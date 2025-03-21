import datetime
import json
import os
import re

import numpy as np
import pandas as pd
import polars as pl
import torch
import tqdm

from mining import estimate, generative

def nansum(x, **kwargs):
    s = np.nansum(x, **kwargs)
    all_nan_rows = np.isnan(x).all(**kwargs)
    if all_nan_rows.any():
        s[np.isnan(x).all(**kwargs)] = np.nan
    return s

class SocialInteractionDataset(torch.utils.data.Dataset):
    def __init__(self, user_df, posts_df, comments_df, post_opinions, comment_opinions, batch=True):
        self.batch = batch

        self.user_df = user_df
        self.posts_df = posts_df
        self.comments_df = comments_df
        self.post_opinions = post_opinions
        self.comment_opinions = comment_opinions

        self.posts_comments_df = self.comments_df.sort_values(['post_id', 'createtime']).groupby('post_id')
        self.user_comments_df = self.comments_df.sort_values(['author_id', 'createtime']).groupby('author_id')

        self.users_df = self.comments_df[['author_id', 'author_name']].drop_duplicates().reset_index()

    def __len__(self):
        return len(self.users_df)

    def __getitem__(self, idx):
        users = self.users_df.iloc[idx]
        comments = pd.concat([self.user_comments_df.get_group(user_id) for user_id in users['author_id']])
        post_ids = comments['post_id'].unique()
        post_comments = pd.concat([self.posts_comments_df.get_group(post_id) for post_id in post_ids])
        # get only level 1 comments
        base_comments = post_comments[post_comments['reply_comment_id'].isna()]
        comment_opinions = self.comment_opinions[base_comments.index.values]

        if self.batch:
            num_users = len(users)
            max_comment_seq = comments[['author_id', 'comment_id']].groupby('author_id').count().max().values[0]
            max_base_comments = base_comments[['post_id', 'comment_id']].groupby('post_id').count().max().values[0]
            
            # add post opinions in front of comment opinions as default option for replying to a post
            padded_reply_opinions = np.zeros((num_users, max_comment_seq, max_base_comments + 1, comment_opinions.shape[1]))
            mask_reply_opinions = np.zeros((num_users, max_comment_seq, max_base_comments + 1,))
            chosen_reply_idx = np.zeros((num_users, max_comment_seq), dtype=np.int64)
            user_comment_opinions = np.zeros((num_users, max_comment_seq, comment_opinions.shape[1]))

            for i, user_id in enumerate(users['author_id']):
                comments = self.user_comments_df.get_group(user_id)
                for j, (comment_idx, comment) in enumerate(comments.iterrows()):
                    # get post
                    post = self.posts_df[self.posts_df['post_id'] == comment['post_id']]
                    assert len(post) == 1
                    post = post.iloc[0]
                    post_opinions = self.post_opinions[post['post_id']]

                    # get all comments for post
                    post_comments = self.posts_comments_df.get_group(comment['post_id'])
                    base_comments = post_comments[post_comments['reply_comment_id'].isna()]
                    comment_opinions = self.comment_opinions[base_comments.index.values]

                    # fill in the comment opinions
                    padded_reply_opinions[i, j, 0, :] = post_opinions
                    padded_reply_opinions[i, j, 1:comment_opinions.shape[0]+1, :] = comment_opinions
                    mask_reply_opinions[i, j, :comment_opinions.shape[0]+1] = 1

                    # fill in the reply comment
                    if not isinstance(comment['reply_comment_id'], str) and np.isnan(comment['reply_comment_id']):
                        reply_comment_idx = 0
                    else:
                        reply_comments = post_comments[post_comments['comment_id'] == comment['reply_comment_id']]
                        assert len(reply_comments) == 1
                        all_comments_reply_idx = reply_comments.index.values[0]
                        # adding 1 because we append the null comment zero vector at the start
                        reply_comment_idx = np.where(base_comments.index.values == all_comments_reply_idx)[0][0] + 1

                    chosen_reply_idx[i, j] = reply_comment_idx

                    # fill in the actual comment
                    user_comment_opinions[i, j, :] = self.comment_opinions[comment_idx]
        else:
            raise NotImplementedError()
            padded_reply_opinions = np.zeros((len(base_comments) + 1, comment_opinions.shape[1]))
            padded_reply_opinions[1:comment_opinions.shape[0]+1,:] = comment_opinions
            mask_reply_opinions = np.ones((len(base_comments) + 1,))

        return {
            'padded_reply_opinions': torch.tensor(padded_reply_opinions),
            'mask_reply_opinions': torch.tensor(mask_reply_opinions),
            'chosen_reply_idx': torch.tensor(chosen_reply_idx),
            'user_comment_opinions': torch.tensor(user_comment_opinions),
        }


class GenerativeInteractionDataset(SocialInteractionDataset):
    def __init__(self, model_creator, num_people=1000, max_timestamp=1000):
        generative_model = model_creator(num_users=num_people)

        time_step = 0
        generative_model.reset()

        for time_step in tqdm.tqdm(range(max_timestamp)):
            generative_model.update(time_step)

        comments_df = generative_model.platform.comments
        posts_df = generative_model.platform.posts
        user_df = generative_model.users

        comments_df = comments_df.reset_index()
        posts_df = posts_df.reset_index()
        user_df = user_df.reset_index()
        user_df = user_df[['index']]

        comments_df.columns = ['comment_id', 'opinions', 'post_id', 'reply_comment_id', 'createtime']
        posts_df.columns = ['post_id', 'opinions', 'createtime']
        user_df.columns = ['author_id']

        post_opinions = generative_model.platform.get_posts_positions()
        comment_opinions = generative_model.platform.get_comments_positions()

        super().__init__(user_df, posts_df, comments_df, post_opinions, comment_opinions)


class TikTokInteractionDataset(SocialInteractionDataset):
    def __init__(self, data_path, batch=True):
        self.batch = batch

        comments_df = pd.read_csv(os.path.join(data_path, 'sample_comments.csv'), dtype={'author_name': str, 'author_id': str, 'comment_id': str, 'video_id': str, 'reply_comment_id': str})
        # fix annoying reply_comment_id stored as exponent format
        comments_df = comments_df.loc[:, ~comments_df.columns.str.contains('^Unnamed')]
        comments_df['createtime'] = pd.to_datetime(comments_df['createtime'])
        
        videos_df = pd.read_csv(os.path.join(data_path, 'sample_videos.csv'), dtype={'author_name': str, 'author_id': str, 'video_id': str, 'share_video_id': str, 'share_video_user_id': str})
        videos_df = videos_df.loc[:, ~videos_df.columns.str.contains('^Unnamed')]
        videos_df['createtime'] = pd.to_datetime(videos_df['createtime'])

        users_df = pd.read_csv(os.path.join(data_path, 'sample_users.csv'), dtype={'author_name': str, 'author_id': str})
        
        # TODO switch from topic probs to embedding
        topic_probs = np.load(os.path.join(data_path, 'probs.npy'))
        comments_topic_probs = topic_probs[:len(comments_df), :]
        video_topic_probs = topic_probs[len(comments_df):, :]

        # get rid of comments where we don't have the replied comment
        comments_df = comments_df[(comments_df['reply_comment_id'].isna()) | (comments_df['reply_comment_id'].isin(comments_df['comment_id'].values))]
        comments_topic_probs = comments_topic_probs[comments_df.index.values]

        video_topic_probs = video_topic_probs[videos_df.index.values]

        comments_df = comments_df.reset_index()
        videos_df = videos_df.reset_index()

        self.__init__(users_df, videos_df, comments_df, video_topic_probs, comments_topic_probs)


class RedditInteractionDataset(SocialInteractionDataset):
    def __init__(self):
        pass


class OpinionTimelineDataset:
    def __init__(
            self, 
            comment_df, 
            stance_columns, 
            all_classifier_profiles, 
            user_mode_attrs=None, 
            aggregation=None, 
            halflife=50.0, 
            min_num_per_stance=None,
            subsample_users=None
        ):
        assert len(comment_df) > 0, "comment_df must not be empty"
        assert 'createtime' in comment_df.columns, "createtime column not found"
        assert 'user_id' in comment_df.columns, "user_id column not found"
        assert 'comment_id' in comment_df.columns, "comment_id column not found"
        assert 'comment' in comment_df.columns, "comment column not found"
        assert 'classifier_idx' in comment_df.columns, "classifier_idx column not found"

        if min_num_per_stance is not None:
            # Create a mask for rows to keep
            mask = pd.Series(True, index=comment_df.index)

            # Iterate over each stance
            for stance in stance_columns:
                # Group by user_id and stance, and filter out groups with less than 10 comments
                small_groups = comment_df[comment_df[stance].notna()].groupby('user_id').filter(lambda x: len(x) < min_num_per_stance).index

                # Update the mask
                mask.loc[small_groups] = False

            # Drop the filtered rows from the DataFrame
            comment_df = comment_df[mask]

        self.comment_df = comment_df
        self.num_opinions = len(stance_columns)
        self.stance_columns = stance_columns
        self.all_classifier_profiles = all_classifier_profiles
        self.num_people = len(self.comment_df['user_id'].unique())
        self.aggregation = aggregation

        self.min_timestamp = self.comment_df['createtime'].min()
        self.max_timestamp = self.comment_df['createtime'].max()
        if isinstance(self.min_timestamp, np.int64):
            self.min_timestamp = int(self.min_timestamp)
            self.max_timestamp = int(self.max_timestamp)

        self.users = self.comment_df['user_id'].to_frame().drop_duplicates().reset_index(drop=True)
        self.subsample_users = subsample_users
        self.users = self.users.sample(n=self.subsample_users) if self.subsample_users is not None else self.users

        if user_mode_attrs is not None:
            self.users = self.users.merge(self.comment_df.groupby('user_id')[user_mode_attrs].agg(lambda x: pd.Series.mode(x)[0]).reset_index(), on='user_id')
        
        self.halflife = halflife

    def get_data(self, start=None, end=None):
        if not start:
            start = self.min_timestamp
        if not end:
            end = self.max_timestamp

        if (isinstance(start, int) or isinstance(start, float) or isinstance(start, datetime.datetime)) and (isinstance(end, int) or isinstance(end, float) or isinstance(end, datetime.datetime)):
            assert start != np.nan or end != np.nan, "start or end cannot be nan"
            comment_df = self.comment_df[(self.comment_df['createtime'] <= end) & (self.comment_df['createtime'] >= start)]

            if len(comment_df) == 0:
                return np.zeros((0, self.num_opinions)), np.zeros((0, self.num_opinions))

            user_comments_df = comment_df.groupby('user_id')
            max_content = user_comments_df.count()['comment_id'].max()

            opinion_sequences = np.full((len(self.users), max_content, self.num_opinions), np.nan)
            opinion_times = np.full((len(self.users), max_content), np.nan)
            classifier_indices = np.zeros((len(self.users), max_content), dtype=np.int64)

            for i, user_id in enumerate(self.users['user_id']):
                try:
                    user_comments = user_comments_df.get_group(user_id)
                except KeyError:
                    continue
                user_comments = user_comments.sort_values('createtime')
                user_opinions = user_comments[self.stance_columns].values
                user_times_series = user_comments['createtime']
                if user_times_series.dtype == np.dtype('datetime64[ns]'):
                    user_times_series = user_times_series.map(lambda t: t.to_pydatetime().timestamp())
                user_times = user_times_series.values
                opinion_sequences[i, :len(user_opinions), :] = user_opinions
                opinion_times[i, :len(user_opinions)] = user_times
                classifier_indices[i, :len(user_opinions)] = user_comments['classifier_idx'].values

            if self.aggregation == "weighted_mean":
                opinion_means, opinion_variances = self._weighted_mean_and_variance(opinion_sequences, classifier_indices)
                return opinion_times, (opinion_means, opinion_variances), self.users
            elif self.aggregation == "inferred_categorical":
                opinion_categorical = estimate.get_inferred_categorical(self, opinion_sequences, classifier_indices)
                return opinion_times, (opinion_categorical,), self.users
            elif self.aggregation == "inferred_beta":
                opinion_beta = estimate.get_inferred_beta(self, opinion_sequences, classifier_indices)
                return opinion_times, (opinion_beta,), self.users
            elif self.aggregation == "inferred_normal":
                opinion_normal, opinion_var = estimate.get_inferred_normal(self, opinion_sequences, classifier_indices)
                return opinion_times, (opinion_normal,opinion_var), self.users
            elif self.aggregation == "weighted_exponential_smoothing":
                days_past = np.full((len(self.users), max_content), np.nan)
                for i, user_id in enumerate(self.users['user_id']):
                    try:
                        user_comments = user_comments_df.get_group(user_id)
                    except KeyError:
                        continue
                    user_comments = user_comments.sort_values('createtime')
                    days_ago = end - user_comments['createtime'].values
                    days_past[i, :len(days_ago)] = days_ago

                opinion_means, opinion_variances = self._weighted_exponential_smoothing(opinion_sequences, days_past, classifier_indices)
                return opinion_times, (opinion_means, opinion_variances), self.users
            elif self.aggregation is None:
                return opinion_times, opinion_sequences, self.users, classifier_indices
            else:
                raise ValueError("Invalid args")
            
        elif hasattr(start, '__iter__') and hasattr(end, '__iter__'):
            assert len(start) == len(end), "start and end must be the same length"
            users_comments = {user_id: user_comments.sort_values('createtime') for user_id, user_comments in self.comment_df.groupby('user_id')}
            if self.aggregation in ["weighted_mean", "weighted_exponential_smoothing"]:
                opinion_timeline = np.zeros((len(end), len(self.users), self.num_opinions))
                opinion_timeline_var = np.zeros((len(end), len(self.users), self.num_opinions))
                precision_weights = None
            elif self.aggregation == "inferred_categorical":
                opinion_timeline = np.zeros((len(end), len(self.users), self.num_opinions, 3))
            elif self.aggregation == "inferred_beta":
                opinion_timeline = np.zeros((len(end), len(self.users), self.num_opinions, 2))
            elif self.aggregation == "inferred_normal":
                opinion_timeline = np.zeros((len(end), len(self.users), self.num_opinions))
                opinion_timeline_var = np.zeros((len(end), len(self.users), self.num_opinions))
            else:
                raise ValueError("Invalid args")

            for idx in tqdm.tqdm(range(len(end))):
                users_time_comments = {user_id: user_comments[(user_comments['createtime'] >= start[idx]) & (user_comments['createtime'] <= end[idx])] for user_id, user_comments in users_comments.items()}
                max_content = max([len(user_comments) for user_comments in users_time_comments.values()])

                opinion_sequences = np.full((len(self.users), max_content, self.num_opinions), np.nan, dtype=np.float16)
                opinion_times = np.full((len(self.users), max_content), np.nan)
                classifier_indices = np.zeros((len(self.users), max_content), dtype=np.int8)
                for i, user_id in enumerate(self.users['user_id']):
                    user_time_comments = users_time_comments[user_id]
                    user_opinions = user_time_comments[self.stance_columns].values
                    user_times = user_time_comments['createtime'].values
                    opinion_sequences[i, :len(user_opinions), :] = user_opinions.astype(np.float16)
                    opinion_times[i, :len(user_opinions)] = user_times
                    classifier_indices[i, :len(user_opinions)] = user_time_comments['classifier_idx'].values.astype(np.int8)

                if self.aggregation in ["weighted_mean", "weighted_exponential_smoothing"]:
                    precision_weights = self._get_prediction_weights(opinion_sequences, classifier_indices)
                        
                    if self.aggregation == "weighted_exponential_smoothing":
                        days_past = np.full((len(self.users), max_content), np.nan)
                        for i, user_id in enumerate(self.users):
                            user_time_comments = users_time_comments[user_id]
                            if isinstance(end[idx], float) and user_time_comments['createtime'].dtype == np.float64:
                                days_ago = np.ceil(end[idx]) - user_time_comments['createtime'].values
                            elif isinstance(end[idx], datetime.datetime) and user_time_comments['createtime'].dtype == np.dtype('datetime64[ns]'):
                                # convert to days
                                days_ago = (pd.to_datetime(end[idx]) - pd.to_datetime(user_time_comments['createtime'])).dt.days.values
                            days_past[i, :len(days_ago)] = days_ago

                        exp_weights = self._get_exp_weights(opinion_sequences, days_past)
                        weights = precision_weights[:, :exp_weights.shape[1], :] * exp_weights
                    else:
                        weights = precision_weights
                    
                    opinion_means, opinion_variances = self._calc_weighted_mean_and_variance(opinion_sequences, weights)
                    opinion_timeline[idx] = opinion_means
                    opinion_timeline_var[idx] = opinion_variances
                elif self.aggregation == "inferred_categorical":
                    opinion_timeline[idx] = estimate.get_inferred_categorical(self, opinion_sequences, classifier_indices)
                elif self.aggregation == "inferred_beta":
                    opinion_timeline[idx] = estimate.get_inferred_beta(self, opinion_sequences, classifier_indices)
                elif self.aggregation == "inferred_normal":
                    mean, variance = estimate.get_inferred_normal(self, opinion_sequences, classifier_indices)
                    opinion_timeline[idx] = mean
                    opinion_timeline_var[idx] = variance

            if self.aggregation in ["weighted_mean", "weighted_exponential_smoothing"]:
                return opinion_times, (opinion_timeline, opinion_timeline_var), self.users
            elif self.aggregation == "inferred_categorical":
                return opinion_times, (opinion_timeline,), self.users
            elif self.aggregation == "inferred_beta":
                return opinion_times, (opinion_timeline,), self.users
            elif self.aggregation == "inferred_normal":
                return opinion_times, (opinion_timeline, opinion_timeline_var), self.users
            else:
                raise ValueError("Invalid args")
        else:
            raise ValueError("Invalid args")

    def get_user_comments(self, user_id):
        return self.comment_df[self.comment_df['user_id'] == user_id]
    
    def _get_prediction_weights(self, opinion_sequences, classifier_indices):
        precisions = np.zeros((len(self.stance_columns), max([len(self.all_classifier_profiles[stance]) for stance in self.stance_columns]), 3))
        precisions = np.nan * precisions
        for stance_idx, stance in enumerate(self.stance_columns):
            for predictor_idx, predictor_id in enumerate(self.all_classifier_profiles[stance]):
                profile = self.all_classifier_profiles[stance][predictor_id]
                precisions[stance_idx, predictor_idx, 0] = profile['against']['precision']
                precisions[stance_idx, predictor_idx, 1] = profile['neutral']['precision']
                precisions[stance_idx, predictor_idx, 2] = profile['favor']['precision']

        weights = np.full(opinion_sequences.shape, np.nan)
        for stance_idx, stance in enumerate(self.stance_columns):
            for user_idx in range(opinion_sequences.shape[0]):
                # Identify valid comments for the current user and stance
                valid_comments = ~np.isnan(opinion_sequences[user_idx, :, stance_idx])

                # Get the predictor IDs for the current user across all comments
                predictor_ids = classifier_indices[user_idx, :]

                # Get the predicted stances for valid comments and adjust for indexing
                predicted_stances = opinion_sequences[user_idx, valid_comments, stance_idx].astype(int) + 1

                # Calculate precision values for valid comments
                precisions_values = precisions[stance_idx, predictor_ids[valid_comments], predicted_stances]

                # Update weights for valid comments
                weights[user_idx, valid_comments, stance_idx] = precisions_values

        return weights

    def _weighted_mean_and_variance(self, opinion_sequences, classifier_indices):
        weights = self._get_prediction_weights(opinion_sequences, classifier_indices)

        return self._calc_weighted_mean_and_variance(opinion_sequences, weights)

    def _calc_weighted_mean_and_variance(self, opinion_sequences, weights):
        weight_sum = nansum(weights, axis=1)

        weighted_mean = np.divide(nansum(opinion_sequences * weights, axis=1), weight_sum, out=np.full(weight_sum.shape, np.nan), where=weight_sum != 0)
        ex2 = np.divide(nansum(opinion_sequences ** 2 * weights, axis=1), weight_sum, out=np.full(weight_sum.shape, np.nan), where=weight_sum != 0)
        n = nansum(~np.isnan(opinion_sequences), axis=1)
        weighted_variance = (ex2 - (weighted_mean ** 2)) * np.divide(n, n - 1, out=np.full(n.shape, np.nan), where=n > 1)
        return weighted_mean, weighted_variance
    
    def _get_exp_weights(self, opinion_sequences, days_past):
        stances = [col.replace('stance_', '') for col in self.stance_columns]
        exp_weights = np.full(opinion_sequences.shape, np.nan)
        alpha = 1 - np.exp(-np.log(2) / self.halflife)

        # Assuming alpha is defined
        one_minus_alpha = 1 - alpha

        for stance_idx, stance in enumerate(stances):
            # Pre-compute a mask for valid comments for all users for this stance
            valid_comments_mask = ~np.isnan(opinion_sequences[:, :, stance_idx])

            # Iterate over users
            for user_idx in range(opinion_sequences.shape[0]):
                valid_comments_indices = np.where(valid_comments_mask[user_idx])[0]

                if valid_comments_indices.size > 0:
                    days_ago = days_past[user_idx, valid_comments_indices]
                    exp_weights[user_idx, valid_comments_indices, stance_idx] = one_minus_alpha ** days_ago

        return exp_weights

    def _weighted_exponential_smoothing(self, opinion_sequences, days_past, classifier_indices):
        precision_weights = self._get_prediction_weights(opinion_sequences, classifier_indices)
        exp_weights = self._get_exp_weights(opinion_sequences, days_past)
        
        weights = precision_weights * exp_weights
        return self._calc_weighted_mean_and_variance(opinion_sequences, weights)


class GenerativeOpinionTimelineDataset(OpinionTimelineDataset):
    def __init__(self, num_people=None, max_timestamp=None, num_opinions=None, generative_model=None, **kwargs):
        if generative_model is None:
            generative_model = generative.SocialGenerativeModel(
                num_users=num_people,
                num_opinions=num_opinions,
            )

            time_step = 0
            generative_model.reset()

            for time_step in tqdm.tqdm(range(max_timestamp)):
                generative_model.update(time_step)

        comments_df = generative_model.platform.comments
        posts_df = generative_model.platform.posts
        user_df = generative_model.users

        if len(comments_df) == 0:
            raise ValueError("No comments generated")

        comments_df = comments_df.reset_index()
        posts_df = posts_df.reset_index()
        user_df = user_df.reset_index()
        user_df = user_df[['index']]

        comments_df.columns = ['comment_id', 'opinions', 'post', 'comment', 'createtime', 'user_id', 'post_id', 'parent_comment_id']
        posts_df.columns = ['post_id', 'opinions', 'createtime', 'user_id']
        user_df.columns = ['user_id']

        num_opinions = generative_model.num_opinions

        stance_columns = [f'stance_{i}' for i in range(num_opinions)]
        comments_df[stance_columns] = np.array([np.array(o) for o in comments_df['opinions']])

        comments_df['classifier_idx'] = 0
        all_classifier_profiles = {}
        for stance in stance_columns:
            all_classifier_profiles[stance] = {}
            all_classifier_profiles[stance][0] = {
                'against': {
                    'precision': 1.0,
                },
                'neutral': {
                    'precision': 1.0,
                },
                'favor': {
                    'precision': 1.0,
                },
                'true_favor': {
                    'predicted_favor': 1,
                    'predicted_against': 0,
                    'predicted_neutral': 0,
                },
                'true_against': {
                    'predicted_favor': 0,
                    'predicted_against': 1,
                    'predicted_neutral': 0,
                },
                'true_neutral': {
                    'predicted_favor': 0,
                    'predicted_against': 0,
                    'predicted_neutral': 1,
                },
            }

        super().__init__(comments_df, stance_columns, all_classifier_profiles, **kwargs)


class SimpleGenerativeOpinionTimelineDataset(OpinionTimelineDataset):
    def __init__(self, user_stance, user_stance_variance, num_data_points, pred_profile_type='perfect', **kwargs):
        if not isinstance(user_stance, float) and user_stance.ndim == 3:
            num_opinions = user_stance.shape[1]
            stances = [f'stance_{i}' for i in range(num_opinions)]
        else:
            num_opinions = 1
            stances = ['stance_0']
        comment_id = 0

        comment_data = []
        for j in range(user_stance.shape[0] if isinstance(user_stance, np.ndarray) and user_stance.ndim > 1 else 1):
            user_id = f"user{j}"
            for i in range(num_data_points):
                if isinstance(user_stance, float):
                    comment_stance = np.random.normal(user_stance, user_stance_variance)
                    stance = stances[0]
                elif isinstance(user_stance, np.ndarray):
                    if user_stance.ndim == 1:
                        comment_stance = np.random.normal(user_stance[i], user_stance_variance)
                        stance = stances[0]
                    elif user_stance.ndim == 2:
                        comment_stance = np.random.normal(user_stance[j, i], user_stance_variance)
                        stance = stances[0]
                    elif user_stance.ndim == 3:
                        stance_idx = np.random.choice(num_opinions)
                        comment_stance = np.random.normal(user_stance[j, stance_idx, i], user_stance_variance[j, stance_idx, i])
                        stance = stances[stance_idx]
                comment_stance = np.clip(comment_stance, -1, 1)
                quantized_comment_stance = 0
                if comment_stance > 1/3:
                    quantized_comment_stance = 1
                elif comment_stance < -1/3:
                    quantized_comment_stance = -1
                comment_data.append({
                    'comment_id': comment_id,
                    'post': f'post{i}',
                    'comment': f'comment{i}',
                    'createtime': i,
                    'user_id': user_id,
                    'post_id': f'post{i}',
                    'parent_comment_id': None,
                    stance: quantized_comment_stance,
                    'classifier_idx': 0,
                })
                comment_id += 1
        comments_df = pd.DataFrame(comment_data)
        stance_columns = stances

        all_classifier_profiles = {}
        for stance in stance_columns:
            all_classifier_profiles[stance] = {}

            if pred_profile_type == 'perfect':
                profile = {
                    'true_favor': {
                        'predicted_favor': 1,
                        'predicted_against': 0,
                        'predicted_neutral': 0,
                    },
                    'true_against': {
                        'predicted_favor': 0,
                        'predicted_against': 1,
                        'predicted_neutral': 0,
                    },
                    'true_neutral': {
                        'predicted_favor': 0,
                        'predicted_against': 0,
                        'predicted_neutral': 1,
                    },
                }
            elif pred_profile_type == 'low_precision':
                profile = {
                    'true_favor': {
                        'predicted_favor': 10,
                        'predicted_against': 1,
                        'predicted_neutral': 7,
                    },
                    'true_against': {
                        'predicted_favor': 1,
                        'predicted_against': 9,
                        'predicted_neutral': 6,
                    },
                    'true_neutral': {
                        'predicted_favor': 7,
                        'predicted_against': 6,
                        'predicted_neutral': 20,
                    },
                }
            elif pred_profile_type == 'low_recall':
                profile = {
                    'true_favor': {
                        'predicted_favor': 7,
                        'predicted_against': 0,
                        'predicted_neutral': 10,
                    },
                    'true_against': {
                        'predicted_favor': 1,
                        'predicted_against': 8,
                        'predicted_neutral': 12,
                    },
                    'true_neutral': {
                        'predicted_favor': 3,
                        'predicted_against': 2,
                        'predicted_neutral': 20,
                    },
                }
            else:
                raise ValueError("Invalid pred_profile_type")

            profile['favor'] = {}
            profile['favor']['precision'] = profile['true_favor']['predicted_favor'] / (profile['true_favor']['predicted_favor'] + profile['true_against']['predicted_favor'] + profile['true_neutral']['predicted_favor'])
            profile['favor']['recall'] = profile['true_favor']['predicted_favor'] / (profile['true_favor']['predicted_favor'] + profile['true_favor']['predicted_against'] + profile['true_favor']['predicted_neutral'])

            profile['against'] = {}
            profile['against']['precision'] = profile['true_against']['predicted_against'] / (profile['true_favor']['predicted_against'] + profile['true_against']['predicted_against'] + profile['true_neutral']['predicted_against'])
            profile['against']['recall'] = profile['true_against']['predicted_against'] / (profile['true_against']['predicted_favor'] + profile['true_against']['predicted_against'] + profile['true_against']['predicted_neutral'])

            profile['neutral'] = {}
            profile['neutral']['precision'] = profile['true_neutral']['predicted_neutral'] / (profile['true_favor']['predicted_neutral'] + profile['true_against']['predicted_neutral'] + profile['true_neutral']['predicted_neutral'])
            profile['neutral']['recall'] = profile['true_neutral']['predicted_neutral'] / (profile['true_neutral']['predicted_favor'] + profile['true_neutral']['predicted_against'] + profile['true_neutral']['predicted_neutral'])

            profile['macro'] = {}
            profile['macro']['precision'] = (profile['favor']['precision'] + profile['against']['precision']) / 2
            profile['macro']['recall'] = (profile['favor']['recall'] + profile['against']['recall']) / 2

            all_classifier_profiles[stance][0] = profile

        super().__init__(comments_df, stance_columns, all_classifier_profiles, **kwargs)


class EnsembleClassifier:
    def __init__(self, metric_profiles):
        self.metric_profiles = metric_profiles

    def predict(self, data_row):

        # TODO change to bayesian model averaging
        # for now we assume that neutral is the most common stance
        # and therefore anything except neutral should be prioritized if the classifier has a high precision for that label
        best_stance = 'neutral'
        highest_precision = 0
        best_classifier_idx = None
        for classifier_idx in self.metric_profiles:
            classification = data_row[f'stance_{classifier_idx}']
            metric_profile = self.metric_profiles[classifier_idx]
            classifier_precision = metric_profile[classification]['precision']
            if classifier_precision > highest_precision:
                highest_precision = classifier_precision
                best_classifier_idx = classifier_idx

                if classification != 'neutral' and classifier_precision > 0.5:
                    best_stance = classification

        return best_stance, best_classifier_idx

class RedditOpinionTimelineDataset(OpinionTimelineDataset):
    def __init__(self, topics_dir_path, **kwargs):

        with open(os.path.join(topics_dir_path, "topic_stances.json"), "r") as f:
            topics_stances = json.load(f)

        # remove french language laws
        topics_stances['topic_stances'] = [topic_stance for topic_stance in topics_stances['topic_stances'] if topic_stance['stances'][0] not in ['french language laws', 'lgbtq rights', 'funding the cbc']]

        comments_df = None
        all_metric_profiles = {}
        for topic_stances in topics_stances['topic_stances']:
            topic_path = os.path.join(topics_dir_path, f"topic_{topic_stances['topics'][0]}")
            for stance in topic_stances['stances']:
                stance_slug = stance.replace(" ", "_")
                stance_path = os.path.join(topic_path, stance_slug)

                if not os.path.exists(stance_path):
                    continue
                
                stance_comments_df = None

                metric_profiles = {}
                for idx, stance_focus in enumerate(['favor', 'against', 'neutral']):
                    best_metric = 0
                    best_metrics = None
                    best_dir_name = None
                    best_num_comments = 0
                    for dir_name in os.listdir(stance_path):
                        metrics_path = os.path.join(stance_path, dir_name, "metrics.json")
                        most_num_comments = 0
                        for filename in os.listdir(os.path.join(stance_path, dir_name)):
                            if not filename.endswith('metrics.json'):
                                num_comments = int(re.search(r'\d+', filename).group())
                                if num_comments > most_num_comments:
                                    most_num_comments = num_comments
                        
                        with open(metrics_path, "r") as f:
                            metrics = json.load(f)
                        metric = metrics[stance_focus]['f0.5']
                        if (metric > best_metric or (metric == best_metric and most_num_comments > best_num_comments)) and metrics[stance_focus]['precision'] > 0.5:
                            best_metric = metric
                            best_metrics = metrics
                            best_dir_name = dir_name
                            best_num_comments = most_num_comments
                    
                    if best_dir_name is None:
                        continue

                    pred_path = os.path.join(stance_path, best_dir_name)
                    # get smallest stance predictions
                    file_name = None
                    for stance_preds_file in os.listdir(pred_path):
                        if stance_preds_file.endswith('comments.parquet.zstd'):
                            this_num_comments = int(re.search(r'\d+', stance_preds_file).group())
                            if this_num_comments == best_num_comments:
                                file_name = stance_preds_file

                    if file_name is None or not os.path.exists(os.path.join(pred_path, file_name)):
                        continue

                    focus_stance_comments_df = pl.read_parquet(
                        os.path.join(pred_path, file_name), 
                        columns=['author', 'id', 'created_utc', 'body', 'body_parent', 'post_all_text', 'stance', 'subreddit']
                    )
                    
                    # rename stance column to stance_{stance}
                    focus_stance_comments_df = focus_stance_comments_df.rename(
                        {
                            f"stance": f"stance_{idx}",
                            "author": "user_id",
                            "id": "comment_id",
                            "created_utc": "createtime",
                            "body": "comment",
                            "body_parent": "parent_comment",
                            "post_all_text": "post",
                            "subreddit": "subreddit"
                        }
                    )

                    metric_profiles[idx] = best_metrics

                    if stance_comments_df is None:
                        stance_comments_df = focus_stance_comments_df
                    else:
                        stance_comments_df = stance_comments_df.join(focus_stance_comments_df.select(['comment_id', f'stance_{idx}']), on='comment_id', how='inner')

                # find best stance prediction for each comment based on metrics for each predictor
                # use a weighted voting method
                stance_comments_df = stance_comments_df.to_pandas()
                weighted_voting_classifier = EnsembleClassifier(metric_profiles)
                stance_comments_df[[stance_slug, 'classifier_idx']] = stance_comments_df.apply(weighted_voting_classifier.predict, axis=1, result_type='expand')
                stance_comments_df = pl.from_pandas(stance_comments_df.drop([f'stance_{idx}' for idx in metric_profiles], axis=1))

                if comments_df is None:
                    comments_df = stance_comments_df
                else:
                    comments_df = pl.concat([comments_df, stance_comments_df], how='diagonal')    

                all_metric_profiles[stance_slug] = metric_profiles

        stance_columns = [col for col in comments_df.columns if col not in ['user_id', 'comment_id', 'createtime', 'comment', 'parent_comment', 'post', 'classifier_idx', 'subreddit']]

        comments_df = comments_df.to_pandas()

        # map neutral to 0, favor to 1, against to -1
        for stance_column in stance_columns:
            def map_stance(stance):
                if stance == "neutral":
                    return 0
                elif stance == "favor":
                    return 1
                elif stance == "against":
                    return -1
                else:
                    return None
            comments_df[stance_column] = comments_df[stance_column].apply(map_stance)

        # round createtime to days, and rescale to start at 0
        comments_df['createtime'] = pd.to_datetime(comments_df['createtime'], unit='s')
        normalize_time = False
        if normalize_time:
            comments_df['createtime'] = comments_df['createtime'].dt.floor('d')
            comments_df['createtime'] = (comments_df['createtime'] - comments_df['createtime'].min()).dt.days

        super().__init__(comments_df, stance_columns, all_metric_profiles, user_mode_attrs=['subreddit'], **kwargs)
