from django.urls import path
from .views import (DogBreed)
 

urlpatterns = [
   path('',DogBreed.as_view(),name="Dog Breed Recognition"),
]