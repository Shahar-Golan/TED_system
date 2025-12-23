import pandas as pd
import numpy as np

# Load the CSV file
csv_path = "data/ted_talks_en.csv"
print("="*80)
print("TED TALKS CSV FILE ANALYSIS")
print("="*80)

try:
    df = pd.read_csv(csv_path)
    
    # Basic Information
    print(f"\n📊 BASIC INFORMATION")
    print(f"{'─'*80}")
    print(f"Total number of entries: {len(df):,}")
    print(f"Total number of columns: {len(df.columns)}")
    print(f"File size: {len(df) * len(df.columns):,} cells")
    
    # Column Headers
    print(f"\n📋 COLUMN HEADERS ({len(df.columns)} columns)")
    print(f"{'─'*80}")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    # Transcript Analysis
    print(f"\n📝 TRANSCRIPT ANALYSIS")
    print(f"{'─'*80}")
    if 'transcript' in df.columns:
        # Remove NaN values for analysis
        transcripts = df['transcript'].dropna()
        
        print(f"Total talks with transcripts: {len(transcripts):,}")
        print(f"Talks missing transcripts: {df['transcript'].isna().sum()}")
        
        # Calculate word counts (tokens)
        word_counts = transcripts.str.split().str.len()
        
        print(f"\n📊 Word Count (Tokens) Statistics:")
        print(f"  Average words: {word_counts.mean():,.0f} words")
        print(f"  Median words: {word_counts.median():,.0f} words")
        print(f"  Shortest transcript: {word_counts.min():,} words")
        print(f"  Longest transcript: {word_counts.max():,} words")
        print(f"  Standard deviation: {word_counts.std():,.0f} words")
        
        # Find the talks with shortest and longest transcripts
        shortest_idx = word_counts.idxmin()
        longest_idx = word_counts.idxmax()
        
        print(f"\n🔍 Transcript Extremes:")
        if 'title' in df.columns and 'speaker_1' in df.columns:
            print(f"  Shortest: \"{df.loc[shortest_idx, 'title']}\" by {df.loc[shortest_idx, 'speaker_1']}")
            print(f"            {word_counts[shortest_idx]:,} words")
            print(f"  Longest:  \"{df.loc[longest_idx, 'title']}\" by {df.loc[longest_idx, 'speaker_1']}")
            print(f"            {word_counts[longest_idx]:,} words")
        
        # Character statistics
        transcript_lengths = transcripts.str.len()
        print(f"\n📏 Character Count Statistics:")
        print(f"  Average characters: {transcript_lengths.mean():,.0f}")
        print(f"  Median characters: {transcript_lengths.median():,.0f}")
        print(f"  Shortest: {transcript_lengths.min():,} characters")
        print(f"  Longest: {transcript_lengths.max():,} characters")
    
    # Views Analysis
    print(f"\n👁️  VIEWS ANALYSIS")
    print(f"{'─'*80}")
    if 'views' in df.columns:
        print(f"Total views across all talks: {df['views'].sum():,}")
        print(f"Average views per talk: {df['views'].mean():,.0f}")
        print(f"Median views: {df['views'].median():,.0f}")
        print(f"Most viewed talk: {df['views'].max():,} views")
        print(f"Least viewed talk: {df['views'].min():,} views")
    
    # Duration Analysis
    print(f"\n⏱️  DURATION ANALYSIS")
    print(f"{'─'*80}")
    if 'duration' in df.columns:
        print(f"Average duration: {df['duration'].mean() / 60:.1f} minutes ({df['duration'].mean():.0f} seconds)")
        print(f"Median duration: {df['duration'].median() / 60:.1f} minutes")
        print(f"Shortest talk: {df['duration'].min() / 60:.1f} minutes")
        print(f"Longest talk: {df['duration'].max() / 60:.1f} minutes")
    
    # Date Range
    print(f"\n📅 DATE RANGE")
    print(f"{'─'*80}")
    if 'recorded_date' in df.columns:
        df['recorded_date'] = pd.to_datetime(df['recorded_date'])
        print(f"Earliest recording: {df['recorded_date'].min().strftime('%Y-%m-%d')}")
        print(f"Latest recording: {df['recorded_date'].max().strftime('%Y-%m-%d')}")
        print(f"Span: {(df['recorded_date'].max() - df['recorded_date'].min()).days} days")
    
    # Language Analysis
    print(f"\n🌍 LANGUAGE AVAILABILITY")
    print(f"{'─'*80}")
    if 'available_lang' in df.columns:
        # Parse the available_lang column (it's a string representation of a list)
        import ast
        lang_counts = []
        for langs in df['available_lang'].dropna():
            try:
                lang_list = ast.literal_eval(langs)
                lang_counts.append(len(lang_list))
            except:
                pass
        
        if lang_counts:
            print(f"Average languages per talk: {np.mean(lang_counts):.1f}")
            print(f"Maximum languages for a single talk: {max(lang_counts)}")
            print(f"Minimum languages for a single talk: {min(lang_counts)}")
    
    # Speakers Analysis
    print(f"\n🎤 SPEAKERS ANALYSIS")
    print(f"{'─'*80}")
    if 'speaker_1' in df.columns:
        unique_speakers = df['speaker_1'].nunique()
        print(f"Total unique speakers: {unique_speakers:,}")
        print(f"Talks per speaker (average): {len(df) / unique_speakers:.2f}")
    
    # Missing Data Summary
    print(f"\n❓ MISSING DATA SUMMARY")
    print(f"{'─'*80}")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    if len(missing_data) > 0:
        print("Columns with missing values:")
        for col, count in missing_data.items():
            percentage = (count / len(df)) * 100
            print(f"  {col}: {count:,} ({percentage:.1f}%)")
    else:
        print("No missing data!")
    
    # Top 5 Most Viewed Talks
    print(f"\n🏆 TOP 5 MOST VIEWED TALKS")
    print(f"{'─'*80}")
    if 'title' in df.columns and 'views' in df.columns and 'speaker_1' in df.columns:
        top_talks = df.nlargest(5, 'views')[['title', 'speaker_1', 'views']]
        for idx, (_, row) in enumerate(top_talks.iterrows(), 1):
            print(f"{idx}. \"{row['title']}\" by {row['speaker_1']}")
            print(f"   Views: {row['views']:,}\n")
    
    print("="*80)
    print("Analysis complete!")
    print("="*80)

except FileNotFoundError:
    print(f"❌ Error: Could not find file at {csv_path}")
except Exception as e:
    print(f"❌ Error analyzing CSV: {e}")
    import traceback
    traceback.print_exc()
