from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import UserCreationForm
from .models import UserProfile, CSVFile
from django.contrib.auth.models import User

# Custom User Creation Form with Role Field
class CustomUserCreationForm(UserCreationForm):
    ROLE_CHOICES = [
        ('analyst', 'Analyst'),
        ('viewer', 'Viewer'),
    ]
    role = forms.ChoiceField(choices=ROLE_CHOICES, initial='viewer', required=True, widget=forms.Select(attrs={'class': 'form-control'}))

    class Meta:
        model = get_user_model()  # Use get_user_model() to fetch the correct User model
        fields = ('username', 'email', 'password1', 'password2')

    def save(self, commit=True):
        user = super().save(commit=False)
        if commit:
            user.save()

            # Create and save UserProfile
            user_profile = UserProfile.objects.create(user=user, role=self.cleaned_data['role'])
            user_profile.save()
            
        return user

# CSV File Upload Form
class CSVUploadForm(forms.ModelForm):
    class Meta:
        model = CSVFile
        fields = ['file']


class EditProfileForm(forms.ModelForm):
    ROLE_CHOICES = [
        ('analyst', 'Analyst'),
        ('viewer', 'Viewer'),
    ]
    role = forms.ChoiceField(choices=ROLE_CHOICES, required=True, widget=forms.Select(attrs={'class': 'form-control'}))

    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name']

    def __init__(self, *args, **kwargs):
        user_profile = kwargs.pop('user_profile', None)
        super().__init__(*args, **kwargs)
        if user_profile:
            self.fields['role'].initial = user_profile.role