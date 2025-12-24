import pandas as pd

# Load the CSV file
df = pd.read_csv('data/ted_talks_en.csv')

# Get talk_ids 12 and 57
talk_ids = [12, 57]

for talk_id in talk_ids:
    # Filter the dataframe for the specific talk_id
    talk = df[df['talk_id'] == talk_id]
    
    if not talk.empty:
        # Get the transcript
        transcript = talk['transcript'].values[0]
        
        # Count words in the transcript
        word_count = len(str(transcript).split())
        
        # Get other info
        title = talk['title'].values[0]
        speaker = talk['speaker_1'].values[0]
        
        print(f"\n{'='*60}")
        print(f"Talk ID: {talk_id}")
        print(f"Title: {title}")
        print(f"Speaker: {speaker}")
        print(f"Transcript length (words): {word_count:,}")
        print(f"{'='*60}")
    else:
        print(f"\nTalk ID {talk_id} not found in the dataset.")

print("\n")
