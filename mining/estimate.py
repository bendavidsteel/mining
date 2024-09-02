import numpy as np
import torch

import gpytorch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro import poutine
import tqdm

class StanceEstimation:
    def __init__(self, all_classifier_profiles):

        def get_predict_probs(true_cat, confusion_profile):
            predict_sum = sum(v for k, v in confusion_profile[true_cat].items())
            predict_probs = torch.tensor([
                confusion_profile[true_cat]["predicted_against"] / predict_sum,
                confusion_profile[true_cat]["predicted_neutral"] / predict_sum,
                confusion_profile[true_cat]["predicted_favor"] / predict_sum,
            ])
            return predict_probs
        
        def get_true_probs(predicted_cat, confusion_profile):
            true_sum = sum(v[predicted_cat] for k, v in confusion_profile.items())
            true_probs = torch.tensor([
                confusion_profile["true_against"][predicted_cat] / true_sum,
                confusion_profile["true_neutral"][predicted_cat] / true_sum,
                confusion_profile["true_favor"][predicted_cat] / true_sum,
            ])
            return true_probs

        self.predictor_confusion_probs = {}
        for stance in all_classifier_profiles:
            
            predict_probs = torch.zeros(len(all_classifier_profiles[stance]), 3, 3)
            true_probs = torch.zeros(len(all_classifier_profiles[stance]), 3, 3)

            assert len(all_classifier_profiles[stance]) == max(all_classifier_profiles[stance].keys()) + 1
            for predictor_id in all_classifier_profiles[stance]:
                classifier_profile = all_classifier_profiles[stance][predictor_id]

                try:
                    confusion_profile = {
                        "true_favor": classifier_profile["true_favor"],
                        "true_against": classifier_profile["true_against"],
                        "true_neutral": classifier_profile["true_neutral"],
                    }
                except KeyError:
                    continue

                for true_idx, true_cat in enumerate(["true_against", "true_neutral", "true_favor"]):
                    try:
                        predict_probs[predictor_id, true_idx, :] = get_predict_probs(true_cat, confusion_profile)
                    except ZeroDivisionError:
                        predict_probs[predictor_id, true_idx, :] = torch.tensor([1/3, 1/3, 1/3])

                for predicted_idx, predicted_cat in enumerate(["predicted_against", "predicted_neutral", "predicted_favor"]):
                    try:
                        true_probs[predictor_id, predicted_idx, :] = get_true_probs(predicted_cat, confusion_profile)
                    except ZeroDivisionError:
                        true_probs[predictor_id, predicted_idx, :] = torch.tensor([1/3, 1/3, 1/3])

            self.predictor_confusion_probs[stance] = {
                'predict_probs': predict_probs,
                'true_probs': true_probs,
            }

    def set_stance(self, stance):
        self.stance = stance

    def model(self, opinion_sequences, predictor_ids, mask, prior=False):
        
        with pyro.plate("batched_data", opinion_sequences.shape[0], dim=-2):
            if prior:
                user_stance_loc = pyro.param("user_stance_loc", torch.tensor(0.0).unsqueeze(0).tile((opinion_sequences.shape[0],1)).to(mask.device), constraint=dist.constraints.interval(-1, 1))
                user_stance_loc_var = pyro.param("user_stance_loc_var", torch.tensor(0.1).unsqueeze(0).tile((opinion_sequences.shape[0],1)).to(mask.device), constraint=dist.constraints.positive)
                user_stance = pyro.sample("user_stance", dist.Normal(user_stance_loc, user_stance_loc_var))
                user_stance_var = pyro.sample(
                    "user_stance_var", 
                    dist.LogNormal(
                        torch.full(((opinion_sequences.shape[0], 1)), 0.1).to(mask.device), 
                        torch.full(((opinion_sequences.shape[0], 1)), 0.2).to(mask.device)
                    )
                )
                
            else:
                user_stance_var = pyro.param("user_stance_var", torch.tensor(0.1).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.positive)
                # # sample stance from the uniform prior
                user_stance = pyro.param("user_stance", torch.tensor(0.0).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.interval(-1, 1))

            # loop over the observed data
            with pyro.plate("observed_data", opinion_sequences.shape[1], dim=-1):
                with poutine.mask(mask=mask):
                    # User creates comments with latent stance
                    # Standard deviation should be half the distance between categories
                    # comment_var = (1/2) ** 2
                    
                    comment_stances = pyro.sample("latent_comment_stance", dist.Normal(user_stance, user_stance_var).expand(opinion_sequences.shape))

                    # Quantize comment stance into 3 categories
                    comment_stance_cats = torch.zeros_like(comment_stances, dtype=torch.int).to(opinion_sequences.device)
                    comment_stance_cats[comment_stances > 1/2] = 2
                    comment_stance_cats[comment_stances < -1/2] = 0
                    comment_stance_cats[(comment_stances >= -1/2) & (comment_stances <= 1/2)] = 1

                    # Get prediction probabilities based on the confusion matrix
                    # Assume self.predictor_confusion_probs is properly vectorized to handle batched inputs
                    predict_probs = self.predictor_confusion_probs[self.stance]['predict_probs'][predictor_ids, comment_stance_cats, :]

                    pyro.sample("predicted_comment_stance", dist.Categorical(probs=predict_probs), obs=opinion_sequences)

    def guide(self, opinion_sequences, predictor_ids, mask, prior=False):
        # comment_stance_var_loc = pyro.param("comment_stance_var_loc", torch.tensor(0.1))
        # comment_stance_var_scale = pyro.param("comment_stance_var_scale", torch.tensor(1.0), constraint=dist.constraints.positive)
        # comment_stance_var = pyro.sample("comment_stance_var", dist.LogNormal(comment_stance_var_loc, comment_stance_var_scale), infer={'is_auxiliary': True})
        # # sample stance from the uniform prior
        
        # loop over the observed data
        with pyro.plate("batched_data", opinion_sequences.shape[0], dim=-2):
            if prior:
                user_stance_q = pyro.param("user_stance_loc_q", torch.tensor(0.0).unsqueeze(0).tile((opinion_sequences.shape[0],1)).to(mask.device), constraint=dist.constraints.interval(-1, 1))
                user_stance_var_q = pyro.param("user_stance_var_q", torch.tensor(0.001).unsqueeze(0).tile((opinion_sequences.shape[0],1)).to(mask.device), constraint=dist.constraints.positive)
                user_stance = pyro.sample("user_stance", dist.Delta(user_stance_q))
                user_stance_var = pyro.sample("user_stance_var", dist.Delta(user_stance_var_q))
            with pyro.plate("observed_data", opinion_sequences.shape[1], dim=-1):
                with poutine.mask(mask=mask):

                    # Get true probabilities from the confusion matrix
                    true_probs = self.predictor_confusion_probs[self.stance]['true_probs'][predictor_ids, opinion_sequences, :]

                    # Sample latent comment stance categories
                    comment_stance_cats = pyro.sample("latent_comment_stance_category", dist.Categorical(probs=true_probs), infer={'is_auxiliary': True})

                    # Determine latent locations based on categories
                    latent_locs = torch.zeros_like(comment_stance_cats, dtype=torch.float).to(opinion_sequences.device)
                    latent_locs[comment_stance_cats == 1] = 0
                    latent_locs[comment_stance_cats == 2] = 1
                    latent_locs[comment_stance_cats == 0] = -1

                    # Sample latent comment stances
                    # comment_stances = pyro.sample("latent_comment_stance", dist.Normal(latent_locs, comment_var))
                    comment_stances = pyro.sample("latent_comment_stance", dist.Delta(latent_locs))

    def beta_model(self, opinion_sequences, predictor_ids, mask, prior=False):
        with pyro.plate("batched_data", opinion_sequences.shape[0], dim=-2):
            if prior:
                alpha = pyro.sample("alpha", dist.LogNormal(torch.full(((opinion_sequences.shape[0], 1)), 2.0), torch.full(((opinion_sequences.shape[0], 1)), 0.2)))
                beta = pyro.sample("beta", dist.LogNormal(torch.full(((opinion_sequences.shape[0], 1)), 2.0), torch.full(((opinion_sequences.shape[0], 1)), 0.2)))
            else:
                alpha = pyro.param("alpha", torch.tensor(10.0).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.positive)
                beta = pyro.param("beta", torch.tensor(10.0).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.positive)

            # loop over the observed data
            with pyro.plate("observed_data", opinion_sequences.shape[1], dim=-1):
                with poutine.mask(mask=mask):
                    # User creates comments with latent stance
                    # Standard deviation should be half the distance between categories
                    # comment_var = (1/2) ** 2
                    comment_stances = pyro.sample("latent_comment_stance", dist.Beta(alpha, beta).expand(opinion_sequences.shape))

                    # Quantize comment stance into 3 categories
                    comment_stance_cats = torch.zeros_like(comment_stances, dtype=torch.int)
                    comment_stance_cats[comment_stances > 2/3] = 2
                    comment_stance_cats[comment_stances < 1/3] = 0
                    comment_stance_cats[(comment_stances >= 1/3) & (comment_stances <= 2/3)] = 1

                    # Get prediction probabilities based on the confusion matrix
                    # Assume self.predictor_confusion_probs is properly vectorized to handle batched inputs
                    predict_probs = self.predictor_confusion_probs[self.stance]['predict_probs'][predictor_ids, comment_stance_cats, :]

                    pyro.sample("predicted_comment_stance", dist.Categorical(probs=predict_probs), obs=opinion_sequences)

    def beta_guide(self, opinion_sequences, predictor_ids, mask, prior=False):
        # loop over the observed data
        # comment_var = pyro.param("comment_var", torch.tensor(0.1).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.interval(0, 0.5**2))

        with pyro.plate("batched_data", opinion_sequences.shape[0], dim=-2):
            if prior:
                alpha_d = pyro.param("alpha_d", torch.tensor(10.0).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.positive)
                beta_d = pyro.param("beta_d", torch.tensor(10.0).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.positive)
                alpha = pyro.sample("alpha", dist.Delta(alpha_d))
                beta = pyro.sample("beta", dist.Delta(beta_d))
            # Standard deviation should be half the distance between categories
            with pyro.plate("observed_data", opinion_sequences.shape[1], dim=-1):
                with poutine.mask(mask=mask):
                    # Get true probabilities from the confusion matrix
                    true_probs = self.predictor_confusion_probs[self.stance]['true_probs'][predictor_ids, opinion_sequences, :]

                    # Sample latent comment stance categories
                    comment_stance_cats = pyro.sample("latent_comment_stance_category", dist.Categorical(probs=true_probs), infer={'is_auxiliary': True})

                    # Determine latent locations based on categories
                    latent_locs = torch.zeros_like(comment_stance_cats, dtype=torch.float)
                    latent_locs[comment_stance_cats == 1] = 0.5
                    latent_locs[comment_stance_cats == 2] = 1
                    latent_locs[comment_stance_cats == 0] = 0

                    # map to support of beta
                    latent_locs = torch.clamp(latent_locs, 1e-6, 1 - 1e-6)
                    # comment_var = torch.min(comment_var, latent_locs * (1 - latent_locs))
                    # alpha = torch.maximum((((1 - latent_locs) / comment_var) - (1 / latent_locs)) * (latent_locs ** 2), torch.tensor(1e-6))
                    # beta = torch.maximum(alpha * ((1 / latent_locs) - 1), torch.tensor(1e-6))

                    # Sample latent comment stances
                    comment_stances = pyro.sample("latent_comment_stance", dist.Delta(latent_locs))
                    # comment_stances = pyro.sample("latent_comment_stance", dist.Beta(alpha, beta))


    def categorical_model(self, opinion_sequences, predictor_ids, mask):
        with pyro.plate("batched_data", opinion_sequences.shape[0], dim=-2):
            user_stance = pyro.param("user_stance", torch.tensor([1/3, 1/3, 1/3]).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.simplex)
            # loop over the observed data
            with pyro.plate("observed_data_sequence", opinion_sequences.shape[1], dim=-1):
                with poutine.mask(mask=mask):
                    # User creates comments with latent stance
                    # Standard deviation should be half the distance between categories
                    # comment_var = (1/2) ** 2
                    comment_stance_cats = pyro.sample(
                        "latent_comment_stance", 
                        dist.Categorical(probs=user_stance.unsqueeze(1)).expand(opinion_sequences.shape)
                    )

                    # Get prediction probabilities based on the confusion matrix
                    # Assume self.predictor_confusion_probs is properly vectorized to handle batched inputs
                    predict_probs = self.predictor_confusion_probs[self.stance]['predict_probs'][predictor_ids, comment_stance_cats, :]

                    pyro.sample("predicted_comment_stance", dist.Categorical(probs=predict_probs), obs=opinion_sequences)


    def categorical_guide(self, opinion_sequences, predictor_ids, mask):
        with pyro.plate("batched_data", opinion_sequences.shape[0], dim=-2):
            # loop over the observed data
            with pyro.plate("observed_data_sequence", opinion_sequences.shape[1], dim=-1):
                with poutine.mask(mask=mask):

                    # Get true probabilities from the confusion matrix
                    true_probs = self.predictor_confusion_probs[self.stance]['true_probs'][predictor_ids, opinion_sequences, :]

                    # Sample latent comment stance categories
                    comment_stance_cats = pyro.sample("latent_comment_stance", dist.Categorical(probs=true_probs))


def get_inferred_categorical(dataset, opinion_sequences, all_classifier_indices):
    estimator = StanceEstimation(dataset.all_classifier_profiles)

    user_stances = np.zeros((opinion_sequences.shape[0], len(dataset.stance_columns), 3))

    # setup the optimizer
    num_steps = 1000
    optim = _get_optimizer(num_steps)
    # for user_idx in range(opinion_sequences.shape[0]):
    for stance_idx, stance_column in enumerate(dataset.stance_columns):

        op_seqs, lengths, all_predictor_ids, max_length = _get_op_seqs(opinion_sequences, stance_idx, all_classifier_indices)
        if max_length == 0:
            continue

        stance_opinion_sequences, classifier_indices, mask = _get_op_seq_with_mask(opinion_sequences, max_length, op_seqs, lengths, all_predictor_ids)

        if stance_column not in estimator.predictor_confusion_probs or len(estimator.predictor_confusion_probs[stance_column]) == 0:
            continue
        estimator.set_stance(stance_column)
        stance_opinion_sequences, classifier_indices, mask = _data_to_torch_device(stance_opinion_sequences, classifier_indices, mask)

        prior = False
        _train_model(estimator.categorical_model, estimator.categorical_guide, optim, stance_opinion_sequences, classifier_indices, mask, prior, num_steps)

        # grab the learned variational parameters
        user_stance = pyro.param("user_stance").detach().numpy()
        user_stances[:, stance_idx] = user_stance

    return user_stances

def get_inferred_beta(dataset, opinion_sequences, all_classifier_indices):
    estimator = StanceEstimation(dataset.all_classifier_profiles)

    user_stances = np.zeros((opinion_sequences.shape[0], len(dataset.stance_columns), 2))

    # setup the optimizer
    num_steps = 1000
    optim = _get_optimizer(num_steps)
    for stance_idx, stance_column in enumerate(dataset.stance_columns):

        op_seqs, lengths, all_predictor_ids, max_length = _get_op_seqs(opinion_sequences, stance_idx, all_classifier_indices)
        if max_length == 0:
            continue

        stance_opinion_sequences, classifier_indices, mask = _get_op_seq_with_mask(opinion_sequences, max_length, op_seqs, lengths, all_predictor_ids)

        if stance_column not in estimator.predictor_confusion_probs or len(estimator.predictor_confusion_probs[stance_column]) == 0:
            continue
        estimator.set_stance(stance_column)
        stance_opinion_sequences, classifier_indices, mask = _data_to_torch_device(stance_opinion_sequences, classifier_indices, mask)

        prior = True
        _train_model(estimator.beta_model, estimator.beta_guide, optim, stance_opinion_sequences, classifier_indices, mask, prior, num_steps)

        # grab the learned variational parameters
        if prior:
            user_stances[:, stance_idx, 0] = pyro.param("alpha_d").detach().numpy().squeeze(-1)
            user_stances[:, stance_idx, 1] = pyro.param("beta_d").detach().numpy().squeeze(-1)
        else:
            user_stances[:, stance_idx, 0] = pyro.param("alpha").detach().numpy().squeeze(-1)
            user_stances[:, stance_idx, 1] = pyro.param("beta").detach().numpy().squeeze(-1)
        # set parameters to nan where no data was available
        user_stances[lengths == 0, stance_idx, 0] = np.nan
        user_stances[lengths == 0, stance_idx, 1] = np.nan

    return user_stances

def _get_optimizer(num_steps):
    
    gamma = 0.1  # final learning rate will be gamma * initial_lr
    initial_lr = 0.01
    lrd = gamma ** (1 / num_steps)
    optim = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})
    return optim

def _train_model(model_func, guide_func, optim, stance_opinion_sequences, classifier_indices, mask, prior, num_steps):
    pyro.clear_param_store()
    # do gradient steps
    svi = SVI(model_func, guide_func, optim, loss=Trace_ELBO())
    losses = []
    for step in range(num_steps):
        loss = svi.step(stance_opinion_sequences, classifier_indices, mask, prior=prior)
        losses.append(loss)

def _get_op_seqs(opinion_sequences, stance_idx, all_classifier_indices):
    op_seqs = []
    lengths = []
    all_predictor_ids = []
    for user_idx in range(opinion_sequences.shape[0]):
        seq = []
        length = 0
        predictor_ids = []
        for i in range(opinion_sequences.shape[1]):
            if not np.isnan(opinion_sequences[user_idx, i, stance_idx]):
                seq.append(opinion_sequences[user_idx, i, stance_idx])
                length += 1
                predictor_ids.append(all_classifier_indices[user_idx, i].astype(int))

        op_seqs.append(np.array(seq))
        lengths.append(length)
        all_predictor_ids.append(np.array(predictor_ids))

    max_length = max(lengths)

    return op_seqs, lengths, all_predictor_ids, max_length

def _get_op_seq_with_mask(opinion_sequences, max_length, op_seqs, lengths, all_predictor_ids):
    stance_opinion_sequences = np.zeros((opinion_sequences.shape[0], max_length))
    classifier_indices = np.zeros((opinion_sequences.shape[0], max_length))
    mask = np.zeros((opinion_sequences.shape[0], max_length))
    for i in range(opinion_sequences.shape[0]):
        stance_opinion_sequences[i, :lengths[i]] = op_seqs[i]
        classifier_indices[i, :lengths[i]] = all_predictor_ids[i]
        mask[i, :lengths[i]] = 1

    return stance_opinion_sequences, classifier_indices, mask

def _data_to_torch_device(stance_opinion_sequences, classifier_indices, device):
    stance_opinion_sequences = torch.tensor(stance_opinion_sequences).int() + 1
    classifier_indices = torch.tensor(classifier_indices).int()
    mask = torch.tensor(mask).bool()

    stance_opinion_sequences = stance_opinion_sequences.to(device)
    classifier_indices = classifier_indices.to(device)
    mask = mask.to(device)

    return stance_opinion_sequences, classifier_indices, mask

def get_inferred_normal(dataset, opinion_sequences, all_classifier_indices):
    estimator = StanceEstimation(dataset.all_classifier_profiles)

    user_stances = np.zeros((opinion_sequences.shape[0], len(dataset.stance_columns)))
    user_stance_vars = np.zeros((opinion_sequences.shape[0], len(dataset.stance_columns)))

    # setup the optimizer
    num_steps = 1000
    optim = _get_optimizer(num_steps)
    if opinion_sequences.shape[0] > 1000:
        device = torch.device("cpu") # currently too big for GPU
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for stance_idx, stance_column in enumerate(dataset.stance_columns):

        op_seqs, lengths, all_predictor_ids, max_length = _get_op_seqs(opinion_sequences, stance_idx, all_classifier_indices)
        if max_length == 0:
            continue

        stance_opinion_sequences, classifier_indices, mask = _get_op_seq_with_mask(opinion_sequences, max_length, op_seqs, lengths, all_predictor_ids)
        
        if stance_column not in estimator.predictor_confusion_probs or len(estimator.predictor_confusion_probs[stance_column]) == 0:
            continue
        estimator.set_stance(stance_column)
        stance_opinion_sequences, classifier_indices, mask = _data_to_torch_device(stance_opinion_sequences, classifier_indices, mask)

        estimator.predictor_confusion_probs[stance_column]['predict_probs'] = estimator.predictor_confusion_probs[stance_column]['predict_probs'].to(device)
        estimator.predictor_confusion_probs[stance_column]['true_probs'] = estimator.predictor_confusion_probs[stance_column]['true_probs'].to(device)

        prior = True
        _train_model(estimator.model, estimator.guide, optim, stance_opinion_sequences, classifier_indices, mask, prior, num_steps)

        # grab the learned variational parameters
        if prior:
            user_stances[:, stance_idx] = pyro.param("user_stance_loc_q").cpu().detach().numpy().squeeze(-1)
            user_stance_vars[:, stance_idx] = pyro.param("user_stance_var_q").cpu().detach().numpy().squeeze(-1)
        else:
            user_stances[:, stance_idx] = pyro.param("user_stance").detach().numpy().squeeze(-1)
            user_stance_vars[:, stance_idx, 1] = pyro.param("user_stance_var").detach().numpy().squeeze(-1)
        # set parameters to nan where no data was available
        user_stances[lengths == 0, stance_idx, 0] = np.nan
        user_stance_vars[lengths == 0, stance_idx, 1] = np.nan

    return user_stances, user_stance_vars

def truncate(times, sequences):
    # hack to get nans
    nan_idxs = torch.where(times != times)[0]
    if len(nan_idxs) == 0:
        return times, sequences
    nan_idx = nan_idxs[0]
    return times[:nan_idx], sequences[:nan_idx]

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gaussian_process(X_norm, y):
    num_users = X_norm.shape[0]
    num_opinions = y.shape[2]
    models = []
    likelihoods = []
    # TODO consider difference likelihood functions
    for i in tqdm.tqdm(range(num_users), "Fitting GPs to users"):
        for j in range(num_opinions):
            train_x, train_y = truncate(X_norm[i,:], y[i,:,j])
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(train_x, train_y, likelihood)
            models.append(model)
            likelihoods.append(likelihood)

    model_list = gpytorch.models.IndependentModelList(*models)
    likelihood_list = gpytorch.likelihoods.LikelihoodList(*likelihoods)

    model_list.train()
    likelihood_list.train()
    optimizer = torch.optim.Adam(model_list.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    mll = gpytorch.mlls.SumMarginalLogLikelihood(likelihood_list, model_list)
    
    losses = []
    training_iter = 50
    for k in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model_list(*model_list.train_inputs)
        # Calc loss and backprop gradients
        loss = -mll(output, model_list.train_targets)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()
        losses.append(loss.item())

    return model_list, likelihood_list, losses

def prep_gp_data(dataset):
    opinion_times, opinion_sequences, users, classifier_indices = dataset.get_data(start=0, end=dataset.max_time_step)
    estimator = StanceEstimation(dataset.all_classifier_profiles)
    estimator.set_stance(dataset.stance_columns[0])
    X = torch.tensor(opinion_times).float()
    max_x = torch.max(torch.nan_to_num(X, 0))
    X_norm = X / max_x
    y = torch.tensor(opinion_sequences).float()

def get_gp_means(dataset, model_list, likelihood_list, X_norm):
    num_users = X_norm.shape[0]
    num_opinions = len(dataset.stance_columns)
    num_timesteps = 500
    # TODO fix for actual timestamps
    timestamps = np.linspace(0, dataset.max_time_step, num_timesteps)
    means = np.full((num_users, num_timesteps, num_opinions), np.nan)
    for i in range(num_users):
        for j in range(num_opinions):
            model = model_list.models[i*num_opinions+j]
            likelihood = likelihood_list.likelihoods[i*num_opinions+j]

            # Get into evaluation (predictive posterior) mode
            model.eval()
            likelihood.eval()

            train_x, train_y = truncate(X_norm[i,:], y[i,:,j])
            x_start = max(torch.min(train_x) - 0.1 * (torch.max(train_x) - torch.min(train_x)), 0.)
            x_end = min(torch.max(train_x) + 0.1 * (torch.max(train_x) - torch.min(train_x)), 1.)
            n_test = int((x_end - x_start) * num_timesteps)
            test_x = torch.linspace(x_start, x_end, n_test)  # test inputs

            # Test points are regularly spaced along [0,1]
            # Make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(test_x))
                mean = observed_pred.mean.numpy()

            start_idx = int(x_start * num_timesteps)
            means[i, start_idx:start_idx+mean.shape[0], j] = mean

    return timestamps, means


def get_inferred_gaussian_process(dataset):
    X_norm, y = prep_gp_data(dataset)
    model_list, likelihood_list, losses = train_gaussian_process(X_norm, y)
    timestamps, means = get_gp_means(dataset, model_list, likelihood_list, X_norm)
    
    return timestamps, means

def plot_gp_fit():
    num_users = len(users)
    num_opinions = len(dataset.stance_columns)
    num_timesteps = 500
    timestamps = np.linspace(0, dataset.max_time_step, num_timesteps)
    means = np.full((num_users, num_timesteps, num_opinions), np.nan)
    for i in range(num_users):
        for j in range(num_opinions):
            model = model_list.models[i*num_opinions+j]
            likelihood = likelihood_list.likelihoods[i*num_opinions+j]

            # Get into evaluation (predictive posterior) mode
            model.eval()
            likelihood.eval()

            train_x, train_y = truncate(X_norm[i,:], y[i,:,j])
            x_start = max(torch.min(train_x) - 0.1 * (torch.max(train_x) - torch.min(train_x)), 0.)
            x_end = min(torch.max(train_x) + 0.1 * (torch.max(train_x) - torch.min(train_x)), 1.)
            n_test = int((x_end - x_start) * num_timesteps)
            test_x = torch.linspace(x_start, x_end, n_test)  # test inputs

            # Test points are regularly spaced along [0,1]
            # Make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(test_x))
                mean = observed_pred.mean.numpy()

            with torch.no_grad():
                # Initialize plot
                f, ax = plt.subplots(1, 1, figsize=(4, 3))

                # Get upper and lower confidence bounds
                lower, upper = observed_pred.confidence_region()
                # Plot training data as black stars
                ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
                # Plot predictive means as blue line
                
                ax.plot(test_x.numpy(), mean, 'b')
                # Shade between the lower and upper confidence bounds
                ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
                ax.set_ylim([-3, 3])
                ax.legend(['Observed Data', 'Mean', 'Confidence'])
                f.savefig(f'./figs/flows/user_{i}_op_{j}_pred.png')

