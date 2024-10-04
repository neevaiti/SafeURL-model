from django.contrib import admin
from django.urls import path
from django.contrib.auth.views import LogoutView
from safeurl.views import home, user_register, user_login, predict, predictions_list, list_models, delete_model, load_model, admin_home, train_model, inspect_model

urlpatterns = [
    path('admin/', admin.site.urls),
    path('admin-dashboard/', admin_home, name='admin_home'),
    path('admin-dashboard/train-model/', train_model, name='train_model'),
    path('admin-dashboard/models/', list_models, name='list_models'),
    path('admin-dashboard/delete-model/<str:version>/', delete_model, name='delete_model'),
    path('admin-dashboard/load-model/<str:version>/', load_model, name='load_model'),
    path('admin-dashboard/inspect-model/<str:version>/', inspect_model, name='inspect_model'),
    path('', home, name='home'),
    path('register/', user_register, name='register'),
    path('login/', user_login, name='login'),
    path('logout/', LogoutView.as_view(next_page='home'), name='logout'),
    path('predict/', predict, name='predict'),
    path('my-predictions/', predictions_list, name='predictions_list'),
]