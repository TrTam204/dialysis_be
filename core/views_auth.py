from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny

from .services import forgot_password, login_user, logout_user, refresh_token, reset_password


@api_view(['POST'])
@permission_classes([AllowAny])
def login_view(request):
    return login_user(request.data)


@api_view(['POST'])
@permission_classes([AllowAny])
def refresh_view(request):
    return refresh_token(request.data)


@api_view(['POST'])
@permission_classes([AllowAny])
def logout_view(request):
    # Stateless JWT: no server-side token storage in Phase 1.
    return logout_user()


@api_view(['POST'])
@permission_classes([AllowAny])
def forgot_password_view(request):
    return forgot_password(request.data.get('email'))


@api_view(['POST'])
@permission_classes([AllowAny])
def reset_password_view(request):
    return reset_password(request.data.get('email'), request.data.get('otp'), request.data.get('password'))
