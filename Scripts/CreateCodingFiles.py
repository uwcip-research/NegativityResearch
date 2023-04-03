import pandas as pd
import pickle
import re

from datetime import datetime
from psycopg2 import sql
from tqdm import tqdm
from collections import defaultdict

# You can install cipwizard with "pip install cipwizard"
from cipwizard.core import sql_statements
from cipwizard.core.util import open_database, to_pandas


coding_header_collaborative = ['Coder', 'Stance', 'Notes', 'URL', 'Quote', 'Original', 'Date',
                            'Quote User', 'Quote User Cluster', 'Quote User Profile', 
                            'Original User', 'Original User Cluster', 'Original User Profile']
cluster_dict = {0: 'Lean-Trump', 2: 'Lean-Socialist', 3: 'Lean-Biden'}


def create_collaborative_coding_file(database, db_config_file, table_name,
                                    start_date, end_date, 
                                    sample_data_file, 
                                    full_data_file, 
                                    sample_num=500, 
                                    cross_cluster_amount=.5,
                                    follower_limit=200000):

    # Open database
    database, cursor = open_database(database, db_config_file)

    # Compose SQL statement to pull data from CIP's servers
    date_statement = sql_statements.date_range([start_date, end_date], equal_after='=')

    sql_statement = sql.SQL("""
    SELECT id, tweet, created_at, user_id, quoted_status_user_id, 
    user_screen_name, quoted_status_user_screen_name, quoting_cluster,
    quoted_cluster 
    FROM {table_name}
    WHERE {date_statement}
    AND quoted_cluster IS NOT NULL -- This filters for only between-influencer quotes.
    AND quoting_cluster NOT IN (1,4) -- Filters out spam clusters
    AND quoted_cluster NOT IN (1,4) -- Filters out spam clusters
    AND user_followers_count >= {follower_limit}
    AND quoted_status_user_followers_count >= {follower_limit}
    AND quoted_status_user_id != user_id -- Remove self-quotes
    """).format(date_statement=date_statement,
                table_name=sql.SQL(table_name),
                follower_limit=sql.SQL(str(follower_limit)))

    # Run command and process results
    print(sql_statement.as_string(cursor))
    cursor.execute(sql_statement)

    print('Query Finished')
    output_data = to_pandas(cursor)
    print(output_data)

    # Save out full data, unformatted
    output_data.to_csv(full_data_file, index=False)

    # Process full data for sampling
    cross_num = sample_num * cross_cluster_amount
    same_num = sample_num * (1 - cross_cluster_amount)

    cross_data = output_data[output_data['quoting_cluster'] != output_data['quoted_cluster']].sample(int(cross_num))
    same_data = output_data[output_data['quoting_cluster'] == output_data['quoted_cluster']].sample(int(same_num))
    process_data = pd.concat([cross_data, same_data]).sort_values('created_at')
    print(process_data)

    output_coding_data = []
    for idx, row in process_data.iterrows():
        quote, original = str.split(row['tweet'], ' QT ')
        output_row = [['Collab.', '', '', f'https://twitter.com/test/status/{row["id"]}',
                    quote, original, row['created_at'], row['user_screen_name'],
                    cluster_dict[row['quoting_cluster']],
                    f'https://twitter.com/intent/user?user_id={row["user_id"]}',
                    row['quoted_status_user_screen_name'], cluster_dict[row['quoted_cluster']],
                    f'https://twitter.com/intent/user?user_id={row["quoted_status_user_id"]}']]
        output_coding_data += output_row

    coding_data = pd.DataFrame(output_coding_data, columns=coding_header_collaborative)
    print(coding_data)
    coding_data.to_csv(sample_data_file, index=False)


    return


def bertopic_modeling(input_csv, output_html):

    df = pd.read_csv(input_csv)

    df.tweet = df.apply(lambda row: re.sub(r"http\S+", "", str(row.tweet)).lower(), 1)
    df.tweet = df.apply(lambda row: re.sub(r'@[^\s]+','@USERNAME', str(row.tweet)).lower(), 1)
    # df.tweet = df.apply(lambda row: " ".join(filter(lambda x:x[0]!="@", row.tweet.split())), 1)
    df.tweet = df.apply(lambda row: " ".join(re.sub("[^a-zA-Z]+", " ", row.tweet).split()), 1)

    print(df)

    from bertopic import BERTopic

    timestamps = df.created_at.to_list()
    tweets = df.tweet.to_list()

    topic_model = BERTopic(verbose=True)
    topics, probs = topic_model.fit_transform(tweets)

    topics_over_time = topic_model.topics_over_time(tweets, timestamps, nr_bins=15)
    fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20)
    fig.write_html(output_html)

    return


if __name__ == '__main__':

    db_config_file = '/home/albeers/Projects/Coronavirus/venus_config.txt'
    database = 'albeers'
    table_name = 'influencer_quote_tweets'

    # Data Sheet, Week 2/10
    if True:
        start_date = datetime(2021, 4, 10)
        end_date = datetime(2021, 6, 26)

        cross_cluster_ratio = 0.75
        sample_num = 500
        output_coding_sheet = '../Datasets/CodingWorksheets/CollaborativeCodingWorksheet_Week2_040223_Sample.csv'
        output_coding_sheet_full = '../Datasets/CodingWorksheets/CollaborativeCodingWorksheet_Week2_040223_FullData.csv'
        create_collaborative_coding_file(database, db_config_file, table_name, start_date, end_date, 
                                        output_coding_sheet, output_coding_sheet_full, sample_num, cross_cluster_ratio)

    # Topics, Week 2/10
    if False:
        input_full_data = '../Datasets/CodingWorksheets/CollaborativeCodingWorksheet_Week2_040223_FullData.csv'
        output_html = '../Datasets/CodingWorksheets/CollaborativeCodingWorksheet_Week2_040223_Topics.html'
        bertopic_modeling(input_full_data, output_html)
