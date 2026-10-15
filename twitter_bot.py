import tweepy
import csv
import argparse
from dotenv import load_dotenv
import os

load_dotenv()

BEARER_TOKEN = os.getenv("TWITTER_BEARER_KEY")

client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

def extract_tweets(hashtag, location, max_results=100):
    if not hashtag.startswith("#"):
        hashtag = "#" + hashtag
    
    query = f"{hashtag} {location}"
    tweets = []
    
    response = client.search_recent_tweets(
        query=query,
        max_results=max_results,
        tweet_fields=['created_at', 'author_id', 'public_metrics'],
        expansions=['author_id'],
        user_fields=['username', 'name', 'location']
    )
    
    if response.data:
        users = {u['id']: u for u in response.includes['users']}
        
        for tweet in response.data:
            user = users.get(tweet.author_id, {})
            tweets.append({
                'username': user.get('username', 'Unknown'),
                'author': user.get('name', 'Unknown'),
                'text': tweet.text,
                'created_at': tweet.created_at,
                'likes': tweet.public_metrics['like_count'],
                'retweets': tweet.public_metrics['retweet_count']
            })
    
    return tweets

def save_to_csv(tweets, filename):
    if not tweets:
        print("No tweets found")
        return
    
    keys = tweets[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(tweets)
    
    print(f"Saved {len(tweets)} tweets to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract tweets by hashtag and location")
    parser.add_argument("--hashtag", type=str, required=True, help="Hashtag to search")
    parser.add_argument("--location", type=str, default="Rourkela", help="Location to search (default: Rourkela)")
    parser.add_argument("--max", type=int, default=100, help="Max tweets (default: 100)")
    parser.add_argument("--output", type=str, help="Output CSV file")
    
    args = parser.parse_args()
    
    tweets = extract_tweets(args.hashtag, args.location, args.max)
    
    if args.output:
        save_to_csv(tweets, args.output)
    else:
        for tweet in tweets:
            print(f"@{tweet['username']}: {tweet['text']}\n")