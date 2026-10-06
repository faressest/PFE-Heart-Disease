from django.urls import path
from . import views 

urlpatterns = [
    path('', views.home, name='home'),
    path('diagnostiquer/', views.predict_heart, name='predict_heart'),
    
]