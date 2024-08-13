import streamlit as st
import Preprocessor
import Helper
from Helper import extract_media
import matplotlib.pyplot as plt
import calendar
import seaborn as sns
import plotly.express as px
import pandas as pd
import os
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import zipfile

# WhatsApp logo and banner
current_dir = os.path.dirname(__file__)
logo_path = os.path.join(current_dir, "whatsapp_logo.png")
banner_path = os.path.join(current_dir, "whatsapp_banner.png")

try:
    st.sidebar.image(logo_path, width=300)
    
except Exception as e:
    print(f"Error opening file: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Logo file exists: {os.path.exists(logo_path)}")
    print(f"Banner file exists: {os.path.exists(banner_path)}")

st.sidebar.title(" 𝕎𝕙𝕒𝕥𝕤𝕒𝕡𝕡 ℂ𝕙𝕒𝕥 𝕊𝕖𝕟𝕥𝕚𝕞𝕖𝕟𝕥 𝔸𝕟𝕒𝕝𝕪𝕤𝕖𝕣 𝔸𝕀 🤖")
st.sidebar.markdown("𝙳𝚎𝚟𝚎𝚕𝚘𝚙𝚎𝚍 𝚋𝚢 𝚂𝙷𝙰𝚂𝙷𝚆𝙰𝚃  𝙼𝙸𝚂𝙷𝚁𝙰", unsafe_allow_html=True)

# Define the tabs variable
tabs = st.tabs(["𝐇𝐨𝐦𝐞", "𝐒𝐭𝐚𝐭𝐬", "𝐌𝐨𝐧𝐭𝐡𝐥𝐲 𝐓𝐢𝐦𝐞𝐥𝐢𝐧𝐞", "𝐃𝐚𝐢𝐥𝐲 𝐓𝐢𝐦𝐞𝐥𝐢𝐧𝐞", "𝐀𝐜𝐭𝐢𝐯𝐢𝐭𝐲", "𝐖𝐨𝐫𝐝𝐂𝐥𝐨𝐮𝐝", "𝐌𝐨𝐬𝐭 𝐂𝐨𝐦𝐦𝐨𝐧 𝐖𝐨𝐫𝐝𝐬", "𝐄𝐦𝐨𝐣𝐢 𝐀𝐧𝐚𝐥𝐲𝐬𝐢𝐬", "𝐒𝐞𝐧𝐭𝐢𝐦𝐞𝐧𝐭 𝐀𝐧𝐚𝐥𝐲𝐬𝐢𝐬", "𝐀𝐰𝐚𝐫𝐝𝐬", "𝐎𝐯𝐞𝐫𝐚𝐥𝐥 𝐃𝐚𝐭𝐚" ,"𝐀𝐛𝐨𝐮𝐭 𝐌𝐞"])

# Home tab
with tabs[0]:
    # Add instructions at the beginning of the app
    if '𝗶𝗻𝘀𝘁𝗿𝘂𝗰𝘁𝗶𝗼𝗻𝘀_𝘀𝗵𝗼𝘄𝗻' not in st.session_state:
        st.write(" 𝑻𝒐  𝒖𝒔𝒆  𝒕𝒉𝒊𝒔  𝒂𝒑𝒑 , 𝑷𝒍𝒆𝒂𝒔𝒆  𝒇𝒐𝒍𝒍𝒐𝒘  𝑻𝒉𝒆𝒔𝒆  𝑺𝒕𝒆𝒑𝒔 : ✍")
        st.write("1. 𝑼𝒑𝒍𝒐𝒂𝒅  𝒂  𝑾𝒉𝒂𝒕𝒔𝑨𝒑𝒑  𝒄𝒉𝒂𝒕  𝒇𝒊𝒍𝒆 (.𝒕𝒙𝒕) , 𝑪𝑺𝑽 𝒇𝒊𝒍𝒆 (.𝒄𝒔𝒗)  𝒐𝒓 𝑬𝒙𝒄𝒆𝒍  𝒇𝒊𝒍𝒆 (.𝒙𝒍𝒔𝒙) 𝒖𝒔𝒊𝒏𝒈 𝒕𝒉𝒆 𝒇𝒊𝒍𝒆 𝒖𝒑𝒍𝒐𝒂𝒅𝒆𝒓 𝒐𝒏 𝒕𝒉𝒆 𝒍𝒆𝒇𝒕 𝒔𝒊𝒅𝒆𝒃𝒂𝒓.")
        st.write("2. 𝑺𝒆𝒍𝒆𝒄𝒕  𝒕𝒉𝒆  𝒖𝒔𝒆𝒓  𝒚𝒐𝒖  𝒘𝒂𝒏𝒕  𝒕𝒐  𝒂𝒏𝒂𝒍𝒚𝒛𝒆  𝒇𝒓𝒐𝒎  𝒕𝒉𝒆  𝒅𝒓𝒐𝒑𝒅𝒐𝒘𝒏  𝒎𝒆𝒏𝒖 .")
        st.write("3. 𝑪𝒍𝒊𝒄𝒌  𝒕𝒉𝒆  '𝑺𝒉𝒐𝒘 𝑨𝒏𝒂𝒍𝒚𝒔𝒊𝒔'  𝒃𝒖𝒕𝒕𝒐𝒏  𝒕𝒐  𝒈𝒆𝒏𝒆𝒓𝒂𝒕𝒆  𝒕𝒉𝒆  𝒔𝒆𝒏𝒕𝒊𝒎𝒆𝒏𝒕 𝒂𝒏𝒂𝒍𝒚𝒔𝒊𝒔.")
        st.write("4. 𝑬𝒙𝒑𝒍𝒐𝒓𝒆  𝒕𝒉𝒆  𝒗𝒂𝒓𝒊𝒐𝒖𝒔 𝒗𝒊𝒔𝒖𝒂𝒍𝒊𝒛𝒂𝒕𝒊𝒐𝒏𝒔  𝒂𝒏𝒅  𝒔𝒕𝒂𝒕𝒊𝒔𝒕𝒊𝒄𝒔  𝒕𝒐  𝒈𝒂𝒊𝒏  𝒊𝒏𝒔𝒊𝒈𝒉𝒕𝒔  𝒊𝒏𝒕𝒐  𝒕𝒉𝒆  𝒄𝒉𝒂𝒕  𝒔𝒆𝒏𝒕𝒊𝒎𝒆𝒏𝒕 .")
        st.write("5. 𝑰𝒏  𝒍𝒂𝒔𝒕  𝒚𝒐𝒖  𝒘𝒊𝒍𝒍  𝑺𝒆𝒆   𝑶𝒗𝒆𝒓𝒂𝒍𝒍  𝑫𝒂𝒕𝒂")
        st.write("6. 𝒀𝒐𝒖  𝒄𝒂𝒏  𝒂𝒍𝒔𝒐  𝒌𝒏𝒐𝒘  𝒎𝒐𝒓𝒆  𝒂𝒃𝒐𝒖𝒕  𝒎𝒆  𝒊𝒏  𝒕𝒉𝒆  '𝑨𝒃𝒐𝒖𝒕 𝑴𝒆'  𝒕𝒂𝒃.")

    # Display WhatsApp banner image at the bottom of instructions
    st.image(banner_path, width=300)
    welcome_messages = {
        "en": "👋 𝚆𝚎𝚕𝚌𝚘𝚖𝚎 𝚝𝚘 𝚝𝚑𝚎 𝚆𝚑𝚊𝚝𝚜𝙰𝚙𝚙 𝙲𝚑𝚊𝚝 𝚂𝚎𝚗𝚝𝚒𝚖𝚎𝚗𝚝 𝙰𝚗𝚊𝚕𝚢𝚣𝚎𝚛! 📊📈 𝚃𝚑𝚒𝚜 𝚝𝚘𝚘𝚕 𝚠𝚒𝚕𝚕 𝚊𝚗𝚊𝚕𝚢𝚣𝚎 𝚢𝚘𝚞𝚛 𝚌𝚑𝚊𝚝 𝚍𝚊𝚝𝚊, 𝚎𝚡𝚝𝚛𝚊𝚌𝚝 𝚒𝚗𝚜𝚒𝚐𝚑𝚝𝚜, 𝚊𝚗𝚍 𝚎𝚟𝚎𝚗 𝚍𝚎𝚝𝚎𝚛𝚖𝚒𝚗𝚎 𝚜𝚎𝚗𝚝𝚒𝚖𝚎𝚗𝚝 𝚞𝚜𝚒𝚗𝚐 𝚊𝚍𝚟𝚊𝚗𝚌𝚎𝚍 𝚖𝚘𝚍𝚎𝚕𝚜. 𝙵𝚎𝚎𝚕 𝚏𝚛𝚎𝚎 𝚝𝚘 𝚞𝚙𝚕𝚘𝚊𝚍 𝚢𝚘𝚞𝚛 𝚌𝚑𝚊𝚝 𝚏𝚒𝚕𝚎, 𝚊𝚗𝚍 𝚕𝚎𝚝'𝚜 𝚍𝚒𝚟𝚎 𝚒𝚗𝚝𝚘 𝚝𝚑𝚎 𝚏𝚊𝚜𝚌𝚒𝚗𝚊𝚝𝚒𝚗𝚐 𝚠𝚘𝚛𝚕𝚍 𝚘𝚏 𝚢𝚘𝚞𝚛 𝚌𝚘𝚗𝚟𝚎𝚛𝚜𝚊𝚝𝚒𝚘𝚗𝚜!",
        "hi": "व्हाट्सएप चैट सेंटीमेंट एनालाइजर में आपका स्वागत है! 📊📈 यह टूल आपके चैट डेटा का विश्लेषण करेगा, अंतर्दृष्टि निकालेगा, और यहां तक ​​कि उन्नत मॉडल का उपयोग करके भावना भी निर्धारित करेगा। बेझिझक अपनी चैट फ़ाइल अपलोड करें, और आइए अपनी बातचीत की आकर्षक दुनिया में उतरें!"
    }
    # Display the welcome messages
    for lang, message in welcome_messages.items():
        st.write(message)
        
    
with tabs[1]:
    uploaded_file = st.sidebar.file_uploader("𝑼𝒑𝒍𝒐𝒂𝒅 𝒕𝒉𝒆 𝑭𝒊𝒍𝒆 ↴ ", type=["txt", "zip"])
    if uploaded_file is not None:
        
        if uploaded_file.type == "text/plain":
            # Process text file
            bytes_data = uploaded_file.getvalue()
            data = bytes_data.decode("utf-8")
            df = Preprocessor.preprocess(data)
        
        elif uploaded_file.type == "application/zip":
            # Process ZIP file
            with zipfile.ZipFile(uploaded_file) as zip_ref:
                zip_ref.extractall()
            # Assuming the ZIP file contains a single text file
            file_name = os.listdir()[0]
            with open(file_name, 'r') as f:
                data = f.read()
            df = Preprocessor.preprocess(data)

        user_list = df['user'].unique().tolist()
        if 'group_notification' in user_list:
            user_list.sort()
        
        if st.sidebar.button("👨‍💻𝑺𝒉𝒐𝒘 𝑨𝒏𝒂𝒍𝒚𝒔𝒊𝒔🌐"):
            selected_user_for_stats = "Overall"  # Display overall data by default

        selected_user_for_stats = st.selectbox("Select a user to analyze stats", user_list, key="stats_user_selectbox")

        if selected_user_for_stats == "Overall":
            num_messages, words, num_media_messages, num_links = Helper.fetch_stats(None, df)
            media_df = extract_media(df)  # Define media_df for the "Overall" case
        else:
            num_messages, words, num_media_messages, num_links = Helper.fetch_stats(selected_user_for_stats, df)
            media_df = extract_media(df)
            media_df = media_df[media_df['Sender'] == selected_user_for_stats]

        # Display stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)

        with col2:
            st.header("Total Words")
            st.title(words)

        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)

        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # Section to display media shared
        st.title('𝑴𝒆𝒅𝒊𝒂 𝑺𝒉𝒂𝒓𝒆𝒅⇲')
        st.dataframe(media_df)

        # Group by sender and type, and count the number of occurrences
        media_counts = media_df.groupby(['Sender', 'Type']).size().reset_index(name='Count')
        st.title('𝑴𝒆𝒅𝒊𝒂 𝑺𝒉𝒂𝒓𝒆𝒅 𝒄𝒐𝒖𝒏𝒕⇲')
        st.dataframe(media_counts)
        
        
# Monthly Timeline 
with tabs[2]:
    st.title("𝑴𝒐𝒏𝒕𝒉𝒍𝒚 𝑻𝒊𝒎𝒆𝒍𝒊𝒏𝒆 ⇲")
    timeline = df.groupby(['year','month_num']).count()['message'].reset_index()

    # Create a new column 'month_name' to display on the x-axis
    timeline['month_name'] = timeline['month_num'].apply(lambda x: calendar.month_name[x])

    # Create a new column 'year_month' to display on the x-axis
    timeline = timeline.assign(year_month=lambda x: x['month_name'] + ' ' + x['year'].astype(str))
    
    # Convert 'year' column to string
    timeline['year'] = timeline['year'].astype(str)  
    
    # Use Plotly Express to create a time series chart
    fig = px.line(timeline, x='year_month', y='message', title='Monthly Message Count')
    fig.update_layout(xaxis_title='Month', yaxis_title='Number of Messages')
    st.plotly_chart(fig)
    st.write(timeline)

    # Add a dropdown to select a user
    user_list = df['user'].unique().tolist()
    user_list.sort()
    selected_user = st.selectbox("Select a user to analyze", user_list, key="monthly_timeline_user_selectbox")

    # Filter the data for the selected user
    user_timeline = df[df['user'] == selected_user].groupby(['year','month_num']).count()['message'].reset_index()

    # Create a new column 'month_name' to display on the x-axis
    user_timeline['month_name'] = user_timeline['month_num'].apply(lambda x: calendar.month_name[x])

    # Create a new column 'year_month' to display on the x-axis
    user_timeline = user_timeline.assign(year_month=lambda x: x['month_name'] + ' ' + x['year'].astype(str))
    
    # Convert 'year' column to string
    user_timeline['year'] = user_timeline['year'].astype(str)

    # Use Plotly Express to create a time series chart
    user_fig = px.line(user_timeline, x='year_month', y='message', title=f'Monthly Message Count for {selected_user}')
    user_fig.update_layout(xaxis_title='Month', yaxis_title='Number of Messages')
    st.plotly_chart(user_fig)
    st.write(user_timeline)
            

# Daily Timeline
with tabs[3]:
    st.title("𝑫𝒂𝒊𝒍𝒚 𝑻𝒊𝒎𝒆𝒍𝒊𝒏𝒆⇲")
    
    # Add a dropdown to select a user
    user_list = df['user'].unique().tolist()
    user_list.sort()
    selected_user_for_daily_timeline = st.selectbox("Select a user to analyze daily timeline", user_list, key="daily_timeline_user_selectbox")

    # Filter the data for the selected user
    daily_timeline = Helper.daily_timeline(selected_user_for_daily_timeline, df)
    daily_timeline['only_date'] = daily_timeline['only_date'].astype(str).str.replace(',', '')

    # Use Plotly Express to create a time series chart
    fig = px.line(daily_timeline, x='only_date', y='message', title='Daily Message Count')
    fig.update_layout(xaxis_title='Date', yaxis_title='Number of Messages')
    st.plotly_chart(fig)
    st.write(daily_timeline)

               
# Activity
with tabs[4]:
    st.title("█▓▒▒░░░𝑨𝒄𝒕𝒊𝒗𝒊𝒕𝒚⇲░░░▒▒▓█")
    
    # Add a dropdown to select a user
    user_list = df['user'].unique().tolist()
    user_list.sort()
    selected_user_for_activity = st.selectbox("Select a user to analyze activity", user_list, key="activity_user_selectbox")
    col1, col2 = st.columns(2)

    with col1:
        st.header("𝑴𝒐𝒔𝒕 𝑩𝒖𝒔𝒚 𝒅𝒂𝒚🙇‍♂️")
        Busy_day = Helper.week_activity_map(selected_user_for_activity, df)
        fig, ax = plt.subplots()
        ax.bar(Busy_day.index, Busy_day.values, color='blue')
        ax.set_xlabel('Day of the Week')  # Add x-axis label
        ax.set_ylabel('Number of Messages')  # Add y-axis label
        plt.xticks(rotation='vertical')

        st.pyplot(fig)

    with col2:
        st.header("𝑴𝒐𝒔𝒕 𝑩𝒖𝒔𝒚 𝑴𝒐𝒏𝒕𝒉🙇")
        Busy_month = Helper.month_activity_map(selected_user_for_activity, df)
        fig, ax = plt.subplots()
        ax.bar(Busy_month.index, Busy_month.values, color='orange')
        ax.set_xlabel('Month')  # Add x-axis label
        ax.set_ylabel('Number of Messages')  # Add y-axis label
        plt.xticks(rotation='vertical')

        st.pyplot(fig)

    st.title("𝐖𝐞𝐞𝐤𝐥𝐲 𝐀𝐜𝐭𝐢𝐯𝐢𝐭𝐲 𝐌𝐚𝐩 ⇲")
    user_heatmap = Helper.activity_heatmap(selected_user_for_activity, df)
    if not user_heatmap.empty:
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)
    else:
        st.write("No data found for the selected user.")

      
with tabs[5]:
    
    # Add a dropdown to select a user
    user_list = df['user'].unique().tolist()
    user_list.sort()
    selected_user_for_wordcloud = st.selectbox("Select a user to analyze wordcloud", user_list, key="wordcloud_user_selectbox")
    temp = df[df['user'] == selected_user_for_wordcloud]
    if not temp.empty:
        wordcloud_image = Helper.create_wordcloud(selected_user_for_wordcloud, temp)
        if wordcloud_image is not None:
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_image)
            st.title(f'WordCloud for {selected_user_for_wordcloud}:')
            st.pyplot(fig)
        else:
            st.write("No data found for the selected user.")
    else:
        st.write("No data found for the selected user.")

# Most Common Words
with tabs[6]:
    
    # Add a dropdown to select a user
    user_list = df['user'].unique().tolist()
    user_list.sort()
    selected_user_for_most_common_words = st.selectbox("Select a user to analyze most common words", user_list, key="most_common_words_user_selectbox")

    # Filter the data for the selected user
    most_common_df = Helper.most_common_words(selected_user_for_most_common_words, df)
    if not most_common_df.empty:
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title(f'Most Common Words for {selected_user_for_most_common_words}:')
        
        # for most common words plot
        ax.set_xlabel('Frequency')  
        ax.set_ylabel('Words')
        st.pyplot(fig)
    else:
        st.write("No data found for the selected user.")
        
        
        
# Emoji Analysis
with tabs[7]:
    
    # Add a dropdown to select a user
    user_list = df['user'].unique().tolist()
    user_list.sort()
    selected_user_for_emoji = st.selectbox("Select a user to analyze emoji usage", user_list, key="emoji_analysis_user_selectbox")

    # Filter the data for the selected user
    user_emoji_df = Helper.emoji_Helper(selected_user_for_emoji, df)
    if not user_emoji_df.empty:
        user_emoji_df.columns = ['Emoji', 'Frequency'] 
        st.title(f'Emoji Analysis for {selected_user_for_emoji}:')
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(user_emoji_df)
        with col2:
            fig = px.bar(user_emoji_df, x='Emoji', y='Frequency', title=f'Emoji Distribution for {selected_user_for_emoji}') 
            fig.update_layout(xaxis_title='Emoji', yaxis_title='Frequency')
            fig.update_layout(font_family='Segoe UI Emoji', font_size=12)
            st.plotly_chart(fig)
    else:
        st.write("No emoji data found for the selected user.")
        
# Sentiment Analysis
with tabs[8]:
    st.title("𝐒𝐞𝐧𝐭𝐢𝐦𝐞𝐧𝐭 𝐀𝐧𝐚𝐥𝐲𝐬𝐢𝐬🤔")
    if uploaded_file is not None:
        sentiment_df = df.copy()
        sentiment_df['sentiment'] = sentiment_df['message'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])
        sentiment_df['sentiment'] = sentiment_df['sentiment'].apply(lambda x: 'Positive' if x > 0.05 else 'Negative' if x < -0.05 else 'Neutral')
        sentiment_df['user'] = df['user']

        # User-level sentiment distribution
        st.header("𝐒𝐞𝐧𝐭𝐢𝐦𝐞𝐧𝐭 𝐃𝐢𝐬𝐭𝐫𝐢𝐛𝐮𝐭𝐢𝐨𝐧 𝐛𝐲 𝐔𝐬𝐞𝐫")
        user_sentiment_df = sentiment_df.groupby('user')['sentiment'].value_counts().reset_index(name='count')
        user_sentiment_df = user_sentiment_df.pivot(index='user', columns='sentiment', values='count').fillna(0)
        fig = px.bar(user_sentiment_df, title='User-level Sentiment Distribution')
        fig.update_layout(xaxis_title='User', yaxis_title='Count')
        st.plotly_chart(fig)

        # User-level sentiment analysis
        st.header("𝐒𝐞𝐧𝐭𝐢𝐦𝐞𝐧𝐭 𝐀𝐧𝐚𝐥𝐲𝐬𝐢𝐬 𝐛𝐲 𝐔𝐬𝐞𝐫")
        user_sentiment_percentages = (user_sentiment_df.div(user_sentiment_df.sum(axis=1), axis=0)) * 100
        st.write(f"User-level Sentiment Percentages:")
        st.write(user_sentiment_percentages)

        # Individual user sentiment analysis
        selected_user = st.selectbox("Select a user to analyze", user_sentiment_df.index, key="sentiment_analysis_user_selectbox")
        individual_sentiment_df = sentiment_df[sentiment_df['user'] == selected_user]
        individual_sentiment_counts = individual_sentiment_df['sentiment'].value_counts()
        individual_sentiment_percentages = (individual_sentiment_counts / individual_sentiment_counts.sum()) * 100
        st.write(f"Sentiment Analysis for {selected_user}:")
        st.write(individual_sentiment_percentages)
    else:
        st.write("No file uploaded. Please upload a file to perform sentiment analysis.")


# Awards
with tabs[9]:
    st.title("𝐀𝐰𝐚𝐫𝐝𝐬 🏆")

    # Define awards
    awards = {
        "Media Lover: The user who shared the most media files",
        "Talk Active: The user with the most messages",
        "Silent Reader: The user with the least messages",
        "Links Sharer: The user who shared the most links",
        "Long Typer: The user with the longest average message length",
        "Mentioner: The user who mentioned others the most",
        "Favourite Domain: The most frequently mentioned domain",
        "Emoji Fan: The user who used the most emojis",
        "Busy Days: The user with the most active days",
        "Quiet Days: The user with the least active days"
    }

    # Display awards
    for award in awards:
        st.write(award)

    # Add a dropdown to select a user
    user_list = df['user'].unique().tolist()
    user_list.sort()
    selected_user_for_awards = st.selectbox("Select a user to analyze awards", user_list, key="awards_user_selectbox")

    # Calculate awards
    if selected_user_for_awards:
        user_stats = Helper.user_stats(selected_user_for_awards, df)

        # Media Lover
        media_lover = "Media Lover: " + user_stats['media_lover']
        st.write(media_lover)

        # Talk Active
        talk_active = "Talk Active: " + user_stats['talk_active']
        st.write(talk_active)

        # Silent Reader
        silent_reader = "Silent Reader: " + user_stats['silent_reader']
        st.write(silent_reader)

        # Links Sharer
        links_sharer = "Links Sharer: " + user_stats['links_sharer']
        st.write(links_sharer)

        # Long Typer
        long_typer = "Long Typer: " + user_stats['long_typer']
        st.write(long_typer)

        # Mentioner
        mentioner = "Mentioner: " + user_stats['mentioner']
        st.write(mentioner)

        # Favourite Domain
        favourite_domain = "Favourite Domain: " + user_stats['favourite_domain']
        st.write(favourite_domain)

        # Emoji Fan
        emoji_fan = "Emoji Fan: " + user_stats['emoji_fan']
        st.write(emoji_fan)

        # Busy Days
        busy_days = "Busy Days: " + user_stats['busy_days']
        st.write(busy_days)

        # Quiet Days
        quiet_days = "Quiet Days: " + user_stats['quiet_days']
        st.write(quiet_days)


# Overall Data
with tabs[10]:
    st.title("𝐎𝐯𝐞𝐫𝐚𝐥𝐥 𝐃𝐚𝐭𝐚 ⇲")
    st.write("This tab provides an overview of the chat data, including the message count for each user, media shared, and overall sentiment distribution.")
    st.write("You can also download the overall data as a ZIP file, which includes a CSV file with the message count for each user and a graph image of the overall sentiment distribution.")
    st.write("To use this tab, follow these steps:")
    st.write("1. Upload a WhatsApp chat file (.txt), CSV file (.csv), or Excel file (.xlsx) using the file uploader on the left sidebar.")
    st.write("2. Wait for the data to be processed and displayed on this tab.")
    st.write("3. Explore the visualizations and statistics to gain insights into the chat data.")
    st.write("4. Click the 'Download Overall Data' button to download the data as a ZIP file.")

    if uploaded_file is not None:
        # Convert 'date' column to date dtype
        df['date'] = df['date'].dt.date

        # Add date input fields
        start_date = st.date_input("Start Date", value=df['date'].min())
        end_date = st.date_input("End Date", value=df['date'].max())

        # Filter dataframe based on date range
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

        overall_df = filtered_df.groupby('user').size().reset_index(name='count').rename(columns={'user': 'User', 'count': 'Message Count'})
        st.write(overall_df)
       

        # Create a bar chart to display the message count for each user
        fig = px.bar(overall_df, x='User', y='Message Count', title='Overall Message Count by User')
        fig.update_layout(xaxis_title='User', yaxis_title='Message Count')
        st.plotly_chart(fig)

        media_df = extract_media(df)
        media_counts = media_df.groupby(['Sender', 'Type']).size().reset_index(name='Count')
        st.write(media_counts)

        # Calculate overall sentiment distribution
        sentiment_df = df.copy()
        sentiment_df['sentiment'] = sentiment_df['message'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])
        sentiment_df['sentiment'] = sentiment_df['sentiment'].apply(lambda x: 'Positive' if x > 0.05 else 'Negative' if x < -0.05 else 'Neutral')
        overall_sentiment_counts = sentiment_df['sentiment'].value_counts()
        overall_sentiment_percentages = (overall_sentiment_counts / overall_sentiment_counts.sum()) * 100
 
        # Create a pie chart to display the overall sentiment distribution
        fig = px.pie(values=overall_sentiment_percentages, names=overall_sentiment_percentages.index, title='Overall Sentiment Distribution')
        st.plotly_chart(fig)

        # Chat Statistics
        st.header("Chat Statistics ⇲ ")
        total_messages = len(df)
        total_words = len(df['message'].str.split().explode().tolist())
        total_media_messages = len(media_df)
        total_links = len(df[df['message'].str.contains('http')])

        # New features
        chat_start_date = df['date'].min()
        chat_end_date = df['date'].max()
        total_missed_calls = len(df[df['message'].str.contains('missed call')])
        total_video_calls = len(df[df['message'].str.contains('video call')])
        total_voice_calls = len(df[df['message'].str.contains('voice call')])
        total_members = len(df['user'].unique())

        deleted_message = len(df[df['message'] == 'This message was deleted'])
        edited_messages = len(df[df['message'].str.contains('edited')])
        shared_contact = len(df[df['message'].str.contains('contact')])
        shared_location = len(df[df['message'].str.contains('location')])

        st.markdown("### Total Deleted Message: ")
        st.write(f"<div class='big-font'>{deleted_message}</div>", unsafe_allow_html=True)
        st.write("---")

        st.markdown("### Total Edited Message: ")
        st.write(f"<div class='big-font'>{edited_messages}</div>", unsafe_allow_html=True)
        st.write("---")

        st.markdown("### Total Contact Shared: ")
        st.write(f"<div class='big-font'>{shared_contact}</div>", unsafe_allow_html=True)
        st.write("---")

        st.markdown("### Total Location Shared: ")
        st.write(f"<div class='big-font'>{shared_location}</div>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(total_messages)

        with col2:
            st.header("Total Words")
            st.title(total_words)

        with col3:
            st.header("Media Shared")
            st.title(total_media_messages)

        with col4:
            st.header("Links Shared")
            st.title(total_links)

        st.header("Additional Chat Statistics ⇲")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Chat Start Date")
            st.write(chat_start_date)

        with col2:
            st.header("Chat End Date")
            st.write(chat_end_date)

        with col3:
            st.header("Total Missed Calls")
            st.title(total_missed_calls)

        with col4:
            st.header("Total Members")
            st.title(total_members)

        st.header("Call Statistics ⇲")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Total Video Calls")
            st.title(total_video_calls)

        with col2:
            st.header("Total Voice Calls")
            st.title(total_voice_calls)

        st.title("Insights ⇲")

        # Top 5 most active users
        st.header("Top 5 Most Active Users")
        top_users = df['user'].value_counts().head(5)
        fig = px.bar(top_users, title='Top 5 Most Active Users')
        fig.update_layout(xaxis_title='User', yaxis_title='Message Count')
        st.plotly_chart(fig)

        # Top 5 most discussed topics
        st.header("Top 5 Most Discussed Topics")
        topic_words = Helper.perform_lda_analysis(df['message'], 5)
        topic_words_df = pd.DataFrame(topic_words, columns=['Topic', 'Words'])
        fig = px.bar(topic_words_df, x='Topic', y='Words', title='Top 5 Most Discussed Topics')
        fig.update_layout(xaxis_title='Topic', yaxis_title='Word Count')
        st.plotly_chart(fig)

        # Chart race: Top 10 users with most messages over time
        st.header("Chart Race: Top 10 Users with Most Messages Over Time")
        user_message_counts = df.groupby(['date', 'user']).size().reset_index(name='count')
        user_message_counts = user_message_counts.sort_values('count', ascending=False).head(10)
        fig = px.line(user_message_counts, x='date', y='count', color='user', title='Chart Race: Top 10 Users with Most Messages Over Time')
        fig.update_layout(xaxis_title='Date', yaxis_title='Message Count')
        st.plotly_chart(fig)

            
        
    
# About Me
with tabs[11]:
    st.title("About Me")

    st.write("My name is Shashwat Mishra, and I'm Completed my Btech in Robotics and Automation.")
    st.write("I'm passionate about machine learning, natural language processing, and data analysis.")
    linkedin_url = "https://www.linkedin.com/in/sm980"  
    st.markdown(f"[LinkedIn Profile]({linkedin_url})", unsafe_allow_html=True)
    st.write("You can find me on LinkedIn by clicking the link above.")