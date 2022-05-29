from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views
urlpatterns = [
    path('', views.index,name='index'),
    path('sendback', views.sendback,name='sendback'),
]
urlpatterns=urlpatterns+static(settings.STATIC_URL,document_root=settings.STATIC_ROOT)