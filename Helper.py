from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from gensim import corpora, models
import re
from collections import Counter
import emoji
extract = URLExtract()


def fetch_stats(selected_user,df):
    if selected_user != 'OverAll':
        df = df[df['user'] == selected_user]

    # Fetch the number of Messages
    num_messages = df.shape[0]

    # Fetch the number of Words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # Fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted> \n'].shape[0]

    # Fetch number of links shared

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))


    return num_messages,len(words),num_media_messages, len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head() 
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(columns= {'user':  'Name', 'count':  'Percent'})
   

    return x, df

def create_wordcloud(selected_user,df):
    

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()


    if selected_user != 'OverAll':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_word(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width = 500, height=500,min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_word)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()


    if selected_user != 'OverAll':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df    

def emoji_Helper(selected_user,df):
    if selected_user != 'OverAll':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        for char in message:
            if char in emoji.EMOJI_DATA:
                emojis.append(char)

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user,df):
    if selected_user != 'OverAll':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year','month_num']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i]+ "-"+ str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user, df):
    # Define 'only_date' as a column name in the df dataframe
    df['only_date'] = df['date'].dt.date
    
    # Filter the dataframe for the selected user
    if selected_user != 'OverAll':
        df = df[df['user'] == selected_user]
    
    # Group the dataframe by 'only_date' and count the number of messages
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    
    return daily_timeline

def week_activity_map(selected_user,df):
    if selected_user != 'OverAll':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):
    if selected_user != 'OverAll':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):
    if selected_user != 'OverAll':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name',columns='period',values='message',aggfunc = 'count').fillna(0)

    return user_heatmap

def extract_media(df):
    media_df = pd.DataFrame(columns=['Sender', 'Type', 'Link/File'])
    for index, row in df.iterrows():
        message = row['message']
        sender = row['user']
        
        # Extract links
        links = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message)
        for link in links:
            media_df = media_df._append({'Sender': sender, 'Type': 'Link', 'Link/File': link}, ignore_index=True)
        
        # Extract videos
        videos = re.findall(r'video|youtube|youtu.be', message)
        for video in videos:
            media_df = media_df._append({'Sender': sender, 'Type': 'Video', 'Link/File': message}, ignore_index=True)
        
        # Extract music
        music = re.findall(r'music|song|audio', message)
        for music_file in music:
            media_df = media_df._append({'Sender': sender, 'Type': 'Music', 'Link/File': message}, ignore_index=True)
        
        # Extract images
        images = re.findall(r'image|photo|jpg|jpeg|png', message)
        for image in images:
            media_df = media_df._append({'Sender': sender, 'Type': 'Image', 'Link/File': message}, ignore_index=True)
        
        # Extract documents
        documents = re.findall(r'doc|docx|pdf|xls|xlsx|ppt|pptx', message)
        for document in documents:
            media_df = media_df._append({'Sender': sender, 'Type': 'Document', 'Link/File': message}, ignore_index=True)
    
    return media_df


def perform_lda_analysis(documents, num_topics):
    # Tokenize the documents
    tokens = [[token.strip() for token in doc.split()] for doc in documents]
    
    # Create a dictionary and corpus
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    
    # Train the LDA model
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    
    # Get the topic words
    def perform_lda_analysis(documents, num_topics):
    # ...
        topic_words = []
        for topic_id, topic_terms in lda_model.print_topics(num_words=5):
            terms = [term.split('*') for term in topic_terms.split('+')]
            terms = [term[1].strip(' ') for term in terms if len(term) > 1]
            topic_words.append((topic_id, ' '.join(terms)))
        return topic_words


def user_stats(selected_user_for_awards, df):
    """
    Calculate user stats for the given user and dataframe.
    """
    user_stats = {
        'media_lover': 'N/A',
        'talk_active': 'N/A',
        'silent_reader': 'N/A',
        'links_sharer': 'N/A',
        'long_typer': 'N/A',
        'mentioner': 'N/A',
        'favourite_domain': 'N/A',
        'emoji_fan': 'N/A',
        'busy_days': 'N/A',
        'quiet_days': 'N/A'
    }

    if selected_user_for_awards:
        user_df = df[df['user'] == selected_user_for_awards]

        # Media Lover
        num_messages, num_words, num_media_messages, num_links = fetch_stats(selected_user_for_awards, df)
        user_stats['media_lover'] = f"{num_media_messages} media files"

        # Talk Active
        message_count = len(user_df)
        user_stats['talk_active'] = f"{message_count} messages"

        # Silent Reader
        if message_count == 0:
            user_stats['silent_reader'] = 'Yes'
        else:
            user_stats['silent_reader'] = 'No'

        # Links Sharer
        link_count = len(user_df[user_df['message'].str.contains('http')])
        user_stats['links_sharer'] = f"{link_count} links"

        # Long Typer
        if message_count > 0:
            avg_message_length = user_df['message'].str.len().mean()
            user_stats['long_typer'] = f"{avg_message_length:.2f} characters per message"

        # Mentioner
        mention_count = user_df['message'].str.count('@').sum()
        user_stats['mentioner'] = f"{mention_count} mentions"

        # Favourite Domain
        if message_count > 0:
            urls = user_df['message'].str.findall(r'https?://([\w.-]+)')
            urls = [url[0] for url in urls if url]
            url_counts = pd.Series(urls).value_counts()
            if len(url_counts) > 0:
                user_stats['favourite_domain'] = url_counts.index[0]

        # Emoji Fan
        emoji_count = sum(1 for msg in user_df['message'] for char in msg if emoji.is_emoji(char))
        user_stats['emoji_fan'] = f"{emoji_count} emojis"

        # Busy Days
        user_dates = user_df['date'].unique()
        user_stats['busy_days'] = f"{len(user_dates)} days"

        # Quiet Days
        if len(user_dates) == 0:
            user_stats['quiet_days'] = 'Yes'
        else:
            user_stats['quiet_days'] = 'No'

    return user_stats