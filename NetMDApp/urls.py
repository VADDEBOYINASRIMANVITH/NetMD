from django.conf.urls import url
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('ingestion', views.ingestion,name='home-page'),
    path('view', views.show_file, name='view'),
    url(r'^view/(?P<id>\d+)/delete', views.file_delete, name='file_delete'),

    path('exploratory', views.exploratory,name='exploratory'),
    path('graph', views.graph, name='graph'),
    # path('visualize', views.visualize),

    path('preprocessing', views.preprocessing,name='preprocessing'),

    path('algo', views.algo, name='algo'),

    path('LR', views.LR),
    path('RF', views.RF),
    path('SVM', views.SVM),
    path('NB', views.NB),
    path('CB', views.CB),
    path('AB', views.AB),
    path('SB', views.SB, name='SB'),
]
