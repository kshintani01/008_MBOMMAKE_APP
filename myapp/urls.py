from django.urls import path
from . import views


urlpatterns = [
    path("", views.index, name="index"),
    path("rules/", views.rules, name="rules"),
    path("ml/", views.ml, name="ml"),
    path("aoai-check/", views.aoai_check, name="aoai_check"),
]