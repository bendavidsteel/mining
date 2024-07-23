import polars as pl

def get_parents(comment_df, submission_df):

    if 'post_permalink' not in comment_df.columns:
        comment_df = comment_df.with_columns(pl.col('permalink').str.extract(r'(\/[^\/]*\/[^\/]*\/[^\/]*\/[^\/]*\/[^\/]*\/)', group_index=1).alias('post_permalink'))

    # get submissions for each context comment
    comment_df = comment_df.join(submission_df.select(pl.col(['permalink', 'title', 'selftext'])), left_on='post_permalink', right_on='permalink', how='left')
    # filter unmatched submissions, probably deleted    
    # comment_df = comment_df.filter(pl.col('title').is_not_null() | pl.col('selftext').is_not_null())

    # t3 is submission, t1 is comment
    # add column indicating this
    comment_df = comment_df.with_columns(pl.col('parent_id').str.contains('t3_').alias('is_base_comment'))

    # get parent comments for each comment
    comment_df = comment_df.with_columns(pl.col('parent_id').str.extract(r't[0-9]{1}_([a-z0-9]*)', group_index=1).alias('parent_specific_id'))
    comment_df = comment_df.join(comment_df.select(pl.col(['id', 'body', 'author', 'permalink'])), left_on='parent_specific_id', right_on='id', how='left', suffix='_parent')

    return comment_df, submission_df