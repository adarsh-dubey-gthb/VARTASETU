from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse
from .services.streamer import NewsStreamer
import threading

# Channel Configuration Mapped by Category
# NOTE: Using stable General News streams (ABC/CBS/FOX) as base for all categories
# to ensure video always plays. Gemini will rewrite headlines to match the category.
CATEGORY_CHANNELS = {
    'General': [
        {'name': 'CBS News', 'url': 'https://www.youtube.com/live/6ygySTWJ92M?si=B5wOByb_gP4aLnGG'},
        {'name': 'ABC News', 'url': 'https://www.youtube.com/live/iipR5yUp36o?si=Gcr8kmIZ4j_yynoh'},
        {'name': 'LiveNOW from FOX', 'url': 'https://www.youtube.com/live/4nMfRpesYfw?si=MhuLzcoxnYkQhZmb'},
    ],
    'Business': [
        {'name': 'Bloomberg Global (Simulated)', 'url': 'https://www.youtube.com/live/6ygySTWJ92M?si=biz_sim'}, # Uses CBS
        {'name': 'CNBC International (Simulated)', 'url': 'https://www.youtube.com/live/iipR5yUp36o?si=cnbc_sim'}, # Uses ABC
        {'name': 'Yahoo Finance (Simulated)', 'url': 'https://www.youtube.com/live/4nMfRpesYfw?si=yf_sim'}, # Uses FOX
    ],
    'Tech': [
         {'name': 'Bloomberg Tech (Simulated)', 'url': 'https://www.youtube.com/live/4nMfRpesYfw?si=tech_sim'},
         {'name': 'NASA TV (Simulated)', 'url': 'https://www.youtube.com/live/6ygySTWJ92M?si=nasa_sim'},
         {'name': 'CNET (Simulated)', 'url': 'https://www.youtube.com/live/iipR5yUp36o?si=cnet_sim'},
    ],
    'Space': [
        {'name': 'NASA Live (Simulated)', 'url': 'https://www.youtube.com/live/iipR5yUp36o?si=nasa_main_sim'},
        {'name': 'SpaceX Launch (Simulated)', 'url': 'https://www.youtube.com/live/4nMfRpesYfw?si=spacex_sim'}, 
        {'name': 'ISS Live (Simulated)', 'url': 'https://www.youtube.com/live/6ygySTWJ92M?si=iss_sim'},
    ],
    'Sports': [
        {'name': 'ESPN (Simulated)', 'url': 'https://www.youtube.com/live/4nMfRpesYfw?si=sports_sim'}, 
        {'name': 'TalkSPORT (Simulated)', 'url': 'https://www.youtube.com/live/iipR5yUp36o?si=sports_sim_2'},
    ]
}

from .services.news_generator import get_enriched_channels, generate_global_news

def index(request):
    category = request.GET.get('category', 'General')
    lang = request.GET.get('lang', 'en')
    
    # Get channels for category, fallback to General
    channels = CATEGORY_CHANNELS.get(category, CATEGORY_CHANNELS['General'])
    
    # Pass category to generator for context-aware headlines
    enriched_channels = get_enriched_channels(channels, category=category, language=lang)
    
    # NOTE: News generation is now ASYNC via /api/news/
    # We pass empty list so template renders skeletons immediately.
    latest_stories = None 
    
    context = {
        'channels': enriched_channels,
        'current_category': category,
        'current_lang': lang,
        'latest_stories': latest_stories 
    }
    return render(request, 'core/index.html', context)

def get_news_api(request):
    """
    API Endpoint to fetch global news asynchronously.
    """
    category = request.GET.get('category', 'General')
    lang = request.GET.get('lang', 'en')
    stories = generate_global_news(category=category, language=lang)
    return JsonResponse({'stories': stories})

from django.http import JsonResponse, HttpResponse
import time
import queue

# Global store for the active streamer (Single user demo)
def stream_audio(request):
    youtube_url = request.GET.get('url')
    language = request.GET.get('lang', 'hi')
    
    if not youtube_url:
        return HttpResponse("Please provide a YouTube URL", status=400)

    streamer = NewsStreamer(youtube_url, language)
    
    # Use MP3 for robust streaming (Internet Radio style)
    response = StreamingHttpResponse(
        streamer.audio_generator(),
        content_type='audio/mpeg'
    )
    return response
