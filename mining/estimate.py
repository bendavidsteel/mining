
from typing import Any, Dict

import gpytorch
import gpytorch.constraints
from gpytorch.likelihoods import OrdinalLikelihood
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro import poutine
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
import torch
from torch import Tensor
import tqdm

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

def get_predictor_confusion_probs(all_classifier_profiles):
    predictor_confusion_probs = {}
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

        predictor_confusion_probs[stance] = {
            'predict_probs': predict_probs,
            'true_probs': true_probs,
        }
    return predictor_confusion_probs

class StanceEstimation:
    def __init__(self, all_classifier_profiles):
        self.predictor_confusion_probs = get_predictor_confusion_probs(all_classifier_profiles)
        
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


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale_loc=1.0, lengthscale_scale=0.5):
        super().__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.LinearMean(input_size=1)
        self.mean_module = gpytorch.means.ConstantMean()
        # TODO set reasonable normal priors
        lengthscale_prior = gpytorch.priors.NormalPrior(lengthscale_loc, lengthscale_scale)
        # TODO figure out which kernel to feed the prior to
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def get_gp_model(train_x, train_y, lengthscale_loc=1.0, lengthscale_scale=0.5):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood, lengthscale_loc=lengthscale_loc, lengthscale_scale=lengthscale_scale)
    return model, likelihood


class DirichletGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes, lengthscale_loc=1.0, lengthscale_scale=0.5):
        super().__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.LinearMean(input_size=1)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size((num_classes,)))
        lengthscale_prior = gpytorch.priors.NormalPrior(lengthscale_loc, lengthscale_scale)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def get_dirichlet_gp_model(train_x, train_y, lengthscale_loc=1.0, lengthscale_scale=0.5):
    if train_y.dtype == torch.float:
        train_y = train_y.int() + 1
    likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(train_y, learn_additional_noise=True)
    model = DirichletGPModel(train_x, likelihood.transformed_targets, likelihood, likelihood.num_classes, lengthscale_loc=lengthscale_loc, lengthscale_scale=lengthscale_scale)
    return model, likelihood

class GPClassificationModel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, lengthscale_loc=1.0, lengthscale_scale=0.5):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(constant_constraint=gpytorch.constraints.Interval(-1, 1))
        lengthscale_prior = gpytorch.priors.LogNormalPrior(loc=torch.log(torch.tensor(lengthscale_loc)), scale=lengthscale_scale)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred
    
    # def predict(self, x):
    #     latent_pred = self(x)
    #     # Apply tanh to constrain the mean
    #     constrained_mean = torch.tanh(latent_pred.mean)
        
    #     # Adjust the covariance matrix using the derivative of tanh
    #     derivative_tanh = 1 - constrained_mean**2
    #     constrained_covar = derivative_tanh.unsqueeze(-1) * latent_pred.covariance_matrix * derivative_tanh.unsqueeze(-2)
        
    #     constrained_pred = gpytorch.distributions.MultivariateNormal(constrained_mean, constrained_covar)
        
    #     return constrained_pred

def inv_probit(x, jitter=1e-3):
    """
    Inverse probit function (standard normal CDF) with jitter for numerical stability.
    
    Args:
        x: Input tensor
        jitter: Small constant to ensure outputs are strictly between 0 and 1
        
    Returns:
        Probabilities between jitter and 1-jitter
    """
    return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0)))) * (1 - 2 * jitter) + jitter

class OrdinalWithErrorLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    def __init__(self, num_classes, all_classifier_profiles):
        super().__init__()
        self.num_classes = num_classes
        self.predictor_confusion_probs = get_predictor_confusion_probs(all_classifier_profiles)
        if torch.cuda.is_available():
            for stance in self.predictor_confusion_probs:
                self.predictor_confusion_probs[stance]['predict_probs'] = self.predictor_confusion_probs[stance]['predict_probs'].to('cuda')
                self.predictor_confusion_probs[stance]['true_probs'] = self.predictor_confusion_probs[stance]['true_probs'].to('cuda')
        
        self.cutpoints = torch.tensor([-0.5, 0.5])
        if torch.cuda.is_available():
            self.cutpoints = self.cutpoints.to('cuda')

        self.register_parameter('sigma', torch.nn.Parameter(torch.tensor(1.0)))
        self.register_constraint('sigma', gpytorch.constraints.Positive())

        self.classifier_ids = None
        self.stance = None

    def set_classifier_ids(self, classifier_ids):
        self.classifier_ids = classifier_ids

    def set_stance(self, stance):
        self.stance = stance

    def forward(self, function_samples: Tensor, *args: Any, data: Dict[str, Tensor] = {}, **kwargs: Any):
        assert self.classifier_ids is not None, "Classifier IDs not set"
        assert self.stance is not None, "Stance not set"

        if isinstance(function_samples, gpytorch.distributions.MultivariateNormal):
            function_samples = function_samples.sample()
        
        # Compute scaled bin edges
        scaled_edges = self.cutpoints / self.sigma
        scaled_edges_left = torch.cat([scaled_edges, torch.tensor([torch.inf], device=scaled_edges.device)], dim=-1)
        scaled_edges_right = torch.cat([torch.tensor([-torch.inf], device=scaled_edges.device), scaled_edges])
        
        # Calculate cumulative probabilities using standard normal CDF (probit function)
        # These represent P(Y ≤ k | F)
        function_samples = function_samples.unsqueeze(-1)
        scaled_edges_left = scaled_edges_left.reshape(1, 1, -1)
        scaled_edges_right = scaled_edges_right.reshape(1, 1, -1)
        probs = inv_probit(scaled_edges_left - function_samples / self.sigma) - inv_probit(scaled_edges_right - function_samples / self.sigma)
        
        # Apply confusion matrix
        predict_probs = torch.einsum('sxc,xco->sxo', probs, self.predictor_confusion_probs[self.stance]['predict_probs'][self.classifier_ids])
        
        return torch.distributions.Categorical(probs=predict_probs)

    # def log_marginal(self, observations, function_dist, *args, **kwargs):
    #     function_samples = function_dist.rsample()
    #     return self.forward(function_samples).log_prob(observations).sum()


def get_ordinal_gp_model(train_x, train_y, all_classifier_profiles, lengthscale_loc=1.0, lengthscale_scale=0.5):
    bin_edges = torch.tensor([-0.5, 0.5])
    likelihood = OrdinalLikelihood(bin_edges)
    # likelihood = OrdinalWithErrorLikelihood(3, all_classifier_profiles)
    model = GPClassificationModel(train_x, lengthscale_loc=lengthscale_loc, lengthscale_scale=lengthscale_scale)
    return model, likelihood

def get_gp_models(X_norm, y, classifier_ids, stance_targets, all_classifier_profiles, lengthscale_loc=1.0, lengthscale_scale=0.5, gp_type='dirichlet'):
    num_users = X_norm.shape[0]
    num_opinions = y.shape[2]
    models = []
    likelihoods = []
    model_map = []
    train_xs = []
    train_ys = []
    # TODO consider difference likelihood functions
    for i in tqdm.tqdm(range(num_users), "Loading data to GPs"):
        for j in range(num_opinions):
            if y[i,:,j].isnan().all():
                continue
            train_x = X_norm[i, ~torch.isnan(y[i,:,j])]
            train_y = y[i, ~torch.isnan(y[i,:,j]), j]
            train_classifier_ids = classifier_ids[i, ~torch.isnan(y[i,:,j])]

            if gp_type in ['dirichlet', 'ordinal']:
                train_y += 1 # convert from -1, 0, 1 to 0, 1, 2
            
            assert (~train_x.isnan().any()) and (~train_y.isnan().any())
            # TODO use a new likelihood considering out y values are classifications
            # https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/GP_Regression_on_Classification_Labels.html
            if gp_type == 'dirichlet':
                model, likelihood = get_dirichlet_gp_model(train_x, train_y, lengthscale_loc=lengthscale_loc, lengthscale_scale=lengthscale_scale)
            elif gp_type == 'normal':
                model, likelihood = get_gp_model(train_x, train_y, lengthscale_loc=lengthscale_loc, lengthscale_scale=lengthscale_scale)
            elif gp_type == 'ordinal':
                model, likelihood = get_ordinal_gp_model(train_x, train_y, all_classifier_profiles, lengthscale_loc=lengthscale_loc, lengthscale_scale=lengthscale_scale)
                likelihood.set_classifier_ids(train_classifier_ids)
                likelihood.set_stance(stance_targets[j])
            models.append(model)
            likelihoods.append(likelihood)
            model_map.append((i, j))
            train_xs.append(train_x)
            train_ys.append(train_y)

    if gp_type in ['normal', 'dirichlet']:
        model_list = gpytorch.models.IndependentModelList(*models)
        likelihood_list = gpytorch.likelihoods.LikelihoodList(*likelihoods)
    elif gp_type == 'ordinal':
        model_list = models
        likelihood_list = likelihoods
    return model_list, likelihood_list, model_map, train_xs, train_ys

def train_gaussian_process(X_norm, y, classifier_ids, stance_targets, all_classifier_profiles, lengthscale_loc=1.0, lengthscale_scale=0.5, gp_type='dirichlet'):
    model_list, likelihood_list, model_map, train_xs, train_ys = get_gp_models(X_norm, y, classifier_ids, stance_targets, all_classifier_profiles, lengthscale_loc=lengthscale_loc, lengthscale_scale=lengthscale_scale, gp_type=gp_type)

    training_iter = 500
    losses = []
    if gp_type == 'normal':
        model_list.train()
        likelihood_list.train()
        optimizer = torch.optim.Adam(model_list.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.SumMarginalLogLikelihood(likelihood_list, model_list)
        for k in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model_list(*model_list.train_inputs)
            # Calc loss and backprop gradients
            loss = -mll(output, model_list.train_targets)
            loss.backward()
            if k % 10 == 0:
                print('Iter %d/%d - Loss: %.3f' % (k + 1, training_iter, loss.item()))
            optimizer.step()
            losses.append(loss.item())
    elif gp_type == 'dirichlet':
        for model, likelihood in zip(model_list.models, likelihood_list.likelihoods):
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for k in range(training_iter):
                optimizer.zero_grad()
                output = model(model.train_inputs[0][:,0])
                loss = -mll(output, likelihood.transformed_targets).sum()
                loss.backward()
                if k % 10 == 0:
                    print('Iter %d/%d - Loss: %.3f' % (k + 1, training_iter, loss.item()))
                optimizer.step()
                losses.append(loss.item())
    elif gp_type == 'ordinal':
        for model, likelihood, model_idxs, train_X, train_y in zip(model_list, likelihood_list, model_map, train_xs, train_ys):
            if torch.cuda.is_available():
                model = model.cuda()
                likelihood = likelihood.cuda()
                train_X = train_X.cuda()
                train_y = train_y.cuda()
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},  # Includes GaussianLikelihood parameters
                {'params': likelihood.parameters()},
            ], lr=0.05)
            training_iter = 5000
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int(training_iter / 10))
            mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y.numel())
            
            with gpytorch.settings.cholesky_max_tries(5):
                best_loss = torch.tensor(float('inf'))
                num_since_best = 0
                num_iters_before_stopping = training_iter // 5
                for k in range(training_iter):
                    optimizer.zero_grad()
                    with gpytorch.settings.variational_cholesky_jitter(1e-4):
                        output = model(train_X)
                        loss = -mll(output, train_y)
                    loss.backward()
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(likelihood.parameters(), max_norm=1.0)
            
                    if k % 50 == 0:
                        print('Iter %d/%d - Loss: %.3f' % (k + 1, training_iter, loss.item()))
                    optimizer.step()
                    scheduler.step(k)
                    losses.append(loss.item())
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        num_since_best = 0
                    else:
                        num_since_best += 1
                    if num_since_best > num_iters_before_stopping:
                        break

    return model_list, likelihood_list, model_map, losses

def nanstd(o,dim):
    return torch.sqrt(
                torch.nanmean(
                    torch.pow( torch.abs(o-torch.nanmean(o,dim=dim).unsqueeze(dim)),2),
                    dim=dim)
                )

def prep_gp_data(dataset):
    opinion_times, opinion_sequences, users, classifier_indices = dataset.get_data()
    estimator = StanceEstimation(dataset.all_classifier_profiles)
    estimator.set_stance(dataset.stance_columns[0])
    X = torch.tensor(opinion_times).float()
    # max_x = torch.max(torch.nan_to_num(X, 0))
    # min_x = torch.min(torch.nan_to_num(X, torch.inf))
    # X_norm = (X - min_x) / (max_x - min_x)
    x_mean = torch.nanmean(X, dim=1)
    x_std = nanstd(X, dim=1)
    X_norm = (X - x_mean.unsqueeze(-1)) / x_std.unsqueeze(-1)
    y = torch.tensor(opinion_sequences).float()
    return X_norm, X, y, classifier_indices

def get_mean_and_confidence_region(model, test_x):
    with gpytorch.settings.cholesky_max_tries(5):
        model_pred = model(test_x)
    if model_pred.loc.is_cuda:
        mean = model_pred.loc.cpu().numpy()
        lower = model_pred.confidence_region()[0].cpu().numpy()
        upper = model_pred.confidence_region()[1].cpu().numpy()
    else:
        mean = model_pred.loc.numpy()
        lower = model_pred.confidence_region()[0].numpy()
        upper = model_pred.confidence_region()[1].numpy()
    return mean, lower, upper

def get_gp_means(dataset, model_list, likelihood_list, model_map, X_norm, X, y):
    num_users = X_norm.shape[0]
    num_opinions = len(dataset.stance_columns)
    num_timesteps = 100
    # TODO fix for actual timestamps
    timestamps = np.full((num_users, num_timesteps), np.nan)
    means = np.full((num_users, num_timesteps, num_opinions), np.nan)
    confidence_region = np.full((num_users, num_timesteps, num_opinions, 2), np.nan)
    for model_idx, (i, j) in enumerate(model_map):
        if hasattr(model_list, 'models'):
            model = model_list.models[model_idx]
            likelihood = likelihood_list.likelihoods[model_idx]
        else:
            model = model_list[model_idx]
            likelihood = likelihood_list[model_idx]

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        train_x_norm = X_norm[i, ~torch.isnan(y[i,:,j])]
        x_norm_start = max(torch.min(train_x_norm), 0.)
        x_norm_end = min(torch.max(train_x_norm), 1.)
        # n_test = int((x_end - x_start) * num_timesteps)
        test_x = torch.linspace(x_norm_start, x_norm_end, num_timesteps)  # test inputs

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            batch_size = 10
            batches = [test_x[i:i+batch_size] for i in range(0, len(test_x), batch_size)]
            mean, lower, upper = get_mean_and_confidence_region(model, batches[0])
            for batch in batches[1:]:
                batch_mean, batch_lower, batch_upper = get_mean_and_confidence_region(model, batch)
                mean = np.concatenate((mean, batch_mean), axis=0)
                lower = np.concatenate((lower, batch_lower), axis=0)
                upper = np.concatenate((upper, batch_upper), axis=0)

        train_x = X[i, ~torch.isnan(y[i,:,j])]
        x_start = torch.min(train_x)
        x_end = torch.max(train_x)

        timestamps[i, :] = np.linspace(x_start, x_end, num_timesteps)
        means[i, :, j] = mean
        confidence_region[i, :, j, 0] = lower
        confidence_region[i, :, j, 1] = upper

    return timestamps, means, confidence_region

def fit_spline_with_ci(X, y, n_knots=4, degree=3, alpha=1e-3, n_bootstraps=1000):
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)
    n_samples = X.shape[0]
    
    # Fit the original model
    model = make_pipeline(SplineTransformer(n_knots=n_knots, degree=degree), Ridge(alpha=alpha))
    model.fit(X, y)
    
    # Prepare bootstrap samples
    bootstrap_indices = np.random.randint(0, n_samples, size=(n_bootstraps, n_samples))
    X_bootstrap = X[bootstrap_indices]
    y_bootstrap = y[bootstrap_indices]
    
    # Fit bootstrap models
    spline_transformer = SplineTransformer(n_knots=n_knots, degree=degree)
    X_spline = spline_transformer.fit_transform(X)
    X_bootstrap_spline = spline_transformer.transform(X_bootstrap.reshape(-1, 1)).reshape(n_bootstraps, n_samples, -1)
    
    # Solve for coefficients
    coef_bootstrap = np.linalg.solve(
        X_bootstrap_spline.transpose(0, 2, 1) @ X_bootstrap_spline + alpha * np.eye(X_spline.shape[1]),
        X_bootstrap_spline.transpose(0, 2, 1) @ y_bootstrap
    )
    
    return model, coef_bootstrap, spline_transformer

def spline_means(X, y, n_bootstraps=1000, alpha=1e-2):
    X = torch.tensor(X).float()
    # max_x = torch.max(torch.nan_to_num(X, 0))
    # min_x = torch.min(torch.nan_to_num(X, torch.inf))
    # X_norm = (X - min_x) / (max_x - min_x)
    x_mean = torch.nanmean(X, dim=1)
    x_std = nanstd(X, dim=1)
    X_norm = (X - x_mean.unsqueeze(-1)) / x_std.unsqueeze(-1)
    y = torch.tensor(y).float()
    models, coef_bootstraps, spline_transformers, model_map = get_splines(X, y, n_bootstraps=n_bootstraps, alpha=alpha)
    timestamps, means, confidence_intervals = get_spline_means(models, coef_bootstraps, spline_transformers, model_map, X_norm, X, y)
    return timestamps, means, confidence_intervals

def get_splines(X_norm, y, n_knots=4, degree=3, alpha=1e-3, n_bootstraps=1000):
    num_users = X_norm.shape[0]
    num_opinions = y.shape[2]
    models = []
    coef_bootstraps = []
    spline_transformers = []
    model_map = []

    for i in tqdm.tqdm(range(num_users), "Fitting splines"):
        for j in range(num_opinions):
            if y[i,:,j].isnan().all():
                continue
            
            train_x = X_norm[i, ~torch.isnan(y[i,:,j])].numpy()
            train_y = y[i, ~torch.isnan(y[i,:,j]), j].numpy()
            
            assert (~np.isnan(train_x).any()) and (~np.isnan(train_y).any())
            
            X_sorted = np.sort(train_x)
            model, coef_bootstrap, spline_transformer = fit_spline_with_ci(
                X_sorted, train_y,
                n_knots=n_knots,
                degree=degree,
                alpha=alpha,
                n_bootstraps=n_bootstraps
            )
            
            models.append(model)
            coef_bootstraps.append(coef_bootstrap)
            spline_transformers.append(spline_transformer)
            model_map.append((i, j))
    
    return models, coef_bootstraps, spline_transformers, model_map

def get_spline_means(model_list, coef_bootstraps, spline_transformers, model_map, X_norm, X, y, ci=95):
    num_users = X_norm.shape[0]
    num_opinions = y.shape[2]
    num_timesteps = 100
    
    timestamps = np.full((num_users, num_timesteps), np.nan)
    means = np.full((num_users, num_timesteps, num_opinions), np.nan)
    confidence_intervals = np.full((num_users, num_timesteps, num_opinions, 2), np.nan)  # New array for CIs

    for model_idx, (i, j) in enumerate(model_map):
        model = model_list[model_idx]
        coef_bootstrap = coef_bootstraps[model_idx]
        spline_transformer = spline_transformers[model_idx]

        train_x_norm = X_norm[i, ~torch.isnan(y[i,:,j])]
        x_norm_start = torch.min(train_x_norm)
        x_norm_end = torch.max(train_x_norm)
        
        test_x = torch.linspace(x_norm_start, x_norm_end, num_timesteps)
        
        # Make predictions
        mean = model.predict(test_x.reshape(-1, 1)).reshape(-1)
        
        # Calculate confidence intervals
        X_spline = spline_transformer.transform(test_x.reshape(-1, 1))
        y_pred_bootstrap = np.einsum('ij,nji->ni', X_spline, coef_bootstrap)
        
        lower_percentile = (100 - ci) / 2
        upper_percentile = 100 - lower_percentile
        y_pred_lower = np.percentile(y_pred_bootstrap, lower_percentile, axis=0)
        y_pred_upper = np.percentile(y_pred_bootstrap, upper_percentile, axis=0)

        train_x = X[i, ~torch.isnan(y[i,:,j])]
        x_start = torch.min(train_x)
        x_end = torch.max(train_x)
        
        timestamps[i, :] = np.linspace(x_start, x_end, num_timesteps)
        means[i, :, j] = mean
        confidence_intervals[i, :, j, 0] = y_pred_lower
        confidence_intervals[i, :, j, 1] = y_pred_upper

    return timestamps, means, confidence_intervals

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

            train_x = X_norm[i, ~torch.isnan(y[i,:,j])]
            train_y = y[i, ~torch.isnan(y[i,:,j]), j]
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

