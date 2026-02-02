from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('stream/', views.stream_audio, name='stream_audio'),
    path('api/news/', views.get_news_api, name='get_news_api'),
]
