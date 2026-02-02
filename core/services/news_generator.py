import os
import yt_dlp
import google.generativeai as genai
import concurrent.futures

# Configure Gemini
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

# Manually load from .env if not found (Local Dev Support)
if not api_key:
    try:
        # Assuming we are in d:\newstts\core\services, go up 2 levels to root
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_path = os.path.join(base_dir, '.env')
        
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('GEMINI_API_KEY=') and not line.startswith('#'):
                        api_key = line.split('=', 1)[1].strip()
                        print("DEBUG: Gemini API Key loaded from .env")
                        break
    except Exception as e:
        print(f"WARNING: Failed to read .env file: {e}")

if api_key:
    genai.configure(api_key=api_key)
else:
    print("WARNING: GEMINI_API_KEY not found. Gemini features will be disabled.")

LANG_MAP = {
    'en': 'English',
    'hi': 'Hindi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'bn': 'Bengali',
    'mr': 'Marathi',
    'ml': 'Malayalam',
    'kn': 'Kannada',
    'gu': 'Gujarati',
    'pa': 'Punjabi',
    'or': 'Odia',
    'as': 'Assamese',
    'ur': 'Urdu'
}

def get_video_info(url):
    """
    Fetches video title and thumbnail from YouTube URL using yt-dlp.
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True, # Fast extraction
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'original_title': info.get('title'),
                'thumbnail': info.get('thumbnail'),
                'url': url
            }
    except Exception as e:
        print(f"Error fetching info for {url}: {e}")
        return {'original_title': 'Live News Stream', 'thumbnail': '', 'url': url}

def generate_headline(video_title, category='General', language='en'):
    """
    Uses Gemini to generate a catchy news headline from a YouTube video title, tailored to the category and language.
    """
    if not api_key:
        return video_title # Fallback

    try:
        # Use user-requested model
        model = genai.GenerativeModel('gemini-3-flash-preview')
        lang_name = LANG_MAP.get(language, 'English')
        
        prompt = f"""
        Act as a professional news editor for a major {category} news outlet.
        Transform this YouTube video title into a concise, engaging, reputable {category} news headline (max 10 words) in {lang_name}.
        
        Rules:
        1. If the category is 'Space', focus on innovation, exploration, or astronomy.
        2. If the category is 'Tech', focus on gadgets, AI, or industry trends.
        3. If the category is 'Sports', focus on the match, athletes, or the game.
        4. Do not use clickbait phrases like "SHOCKING" or ALL CAPS.
        5. Make it sound authoritative.
        6. Output ONLY the headline in {lang_name}.
        
        Video Title: "{video_title}"
        
        Headline:
        """
        response = model.generate_content(prompt)
        return response.text.strip().replace('"', '')
    except Exception as e:
        print(f"Gemini Error: {e}")
        return video_title

def process_channel(channel, category='General', language='en'):
    """
    Process a single channel dict: fetch info -> generate headline.
    """
    # 1. Get YouTube Info
    info = get_video_info(channel['url'])
    
    # 2. Generate Headline
    headline = generate_headline(info['original_title'], category, language)
    
    # 3. Combine
    return {
        'name': channel['name'],
        'url': channel['url'],
        'thumbnail': info['thumbnail'],
        'headline': headline,
        'original_title': info['original_title']
    }

import time

def get_enriched_channels(channels, category='General', language='en'):
    """
    Takes a list of simple channel dicts ({name, url}) and returns enriched data 
    (thumbnail, generated_headline) sequentially to avoid 429 Quota errors.
    """
    enriched_channels = []
    # Use 1 worker to prevent burst rate limit (5 RPM)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future_to_channel = {executor.submit(process_channel, channel, category, language): channel for channel in channels}
        for future in concurrent.futures.as_completed(future_to_channel):
            try:
                data = future.result()
                enriched_channels.append(data)
                # Polite delay between requests
                time.sleep(1.5) 
            except Exception as e:
                print(f"Error processing channel: {e}")
                channel = future_to_channel[future]
                channel['headline'] = channel['name'] # Fallback
                channel['thumbnail'] = ''
                enriched_channels.append(channel)
    
    return enriched_channels

def generate_global_news(category='General', language='en'):
    """
    Generates a list of 6 realistic 'breaking news' stories for the homepage.
    Returns a list of dicts: [{'headline': ..., 'summary': ..., 'category': ...}]
    """
    lang_name = LANG_MAP.get(language, 'English')
    
    if not api_key:
        # Fallback static content if no API key
        return [
            {'headline': 'No API Key Configured', 'summary': 'Please set GEMINI_API_KEY in .env', 'category': 'System'},
            {'headline': 'Static Fallback News', 'summary': 'News generation is disabled.', 'category': 'System'}
        ] * 3

    try:
        # Revert to user-requested model
        model = genai.GenerativeModel('gemini-3-flash-preview')
        prompt = f"""
        Act as a professional news editor.
        Generate 12 distinct, realistic, and engaging "Breaking News" stories for a {category} News Homepage in {lang_name}.
        
        The stories should cover diverse topics appropriately (e.g., if General: Politics, Tech, Science, World).
        
        Output strictly in this pipe-separated format (one story per line):
        Headline | Short Summary (15-20 words) | Sub-Category (e.g. Politics, Tech)
        
        Rules:
        1. Both Headline and Summary MUST be in {lang_name}.
        2. Sub-Category can be in English or {lang_name} (English specific is better for styling but {lang_name} is fine).
        
        Example (if English):
        Global Markets Rally on Tech Optimism | Major indices hit record highs as AI adoption accelerates across sectors. | Finance
        
        """
        
        response = model.generate_content(prompt)
        stories = []
        
        # Parse the pipe-separated response
        lines = response.text.strip().split('\n')
        for line in lines:
            parts = line.split('|')
            if len(parts) >= 3:
                stories.append({
                    'headline': parts[0].strip(),
                    'summary': parts[1].strip(),
                    'category': parts[2].strip()
                })
        
        # Ensure we have at least some stories
        if not stories:
            raise Exception("Failed to parse stories")
            
        return stories

    except Exception as e:
        print(f"Global News Gen Error: {e}")
        # Robust Fallback - Return these immediately on error so the page doesn't hang/show loading forever
        return [
            {'headline': f'{category} News Update Pending', 'summary': 'Real-time updates are syncronizing.', 'category': 'World'},
            {'headline': 'Market Watch', 'summary': 'Trading remains steady as investors await new data.', 'category': 'Business'},
            {'headline': 'Tech Summit Announced', 'summary': 'Leaders gather to discuss the future of AI safety.', 'category': 'Tech'},
            {'headline': 'Climate Accord Talks', 'summary': 'Nations aim for ambitious new targets this week.', 'category': 'Environment'},
            {'headline': 'Sports Championship Preview', 'summary': 'Fans gear up for the season finale this weekend.', 'category': 'Sports'},
            {'headline': 'Space Exploration Milestone', 'summary': 'New mission launch date confirmed by agencies.', 'category': 'Space'}
        ]
