from django.urls import path
from . import views

urlpatterns = [
    path("query", views.query, name="query"),
    path("ingest/upload", views.ingest_upload, name="ingest_upload"),
]
