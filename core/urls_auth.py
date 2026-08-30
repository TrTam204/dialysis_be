from django.urls import path

from .views_auth import forgot_password_view, login_view, logout_view, refresh_view, reset_password_view

urlpatterns = [
    path('login/', login_view, name='login'),
    path('refresh/', refresh_view, name='token_refresh'),
    path('logout/', logout_view, name='logout'),
    path('forgot-password/', forgot_password_view, name='forgot_password'),
    path('reset-password/', reset_password_view, name='reset_password'),
]
