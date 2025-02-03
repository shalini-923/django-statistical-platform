from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

app_name = 'csv_visualizer'

urlpatterns = [
    path('', views.home, name='home'), 
    path('upload/', views.upload_csv, name='upload_csv'),
    path('visualize/', views.visualize_csv, name='visualize_csv'),
    path('download_csv/<int:file_id>/', views.download_csv_file, name='download_csv_file'),
    path('download_analysis_pdf/<int:analysis_id>/', views.download_analysis_pdf, name='download_analysis_pdf'),
    path('history/', views.user_upload_history, name='user_upload_history'),
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login_view'),
    path('delete_csv/<int:file_id>/', views.delete_csv, name='delete_csv'),
    path('logout/', views.custom_logout, name='custom_logout'), 
    path('visualize/', views.visualize_csv, name='visualize_csv_no_id'), 
    path('visualize/<int:file_id>/', views.visualize_csv, name='visualize_csv'),  
    path('chartjs_visualization/<int:file_id>/', views.chartjs_visualization, name='chartjs_visualization'),
    path('chartjs_visualization/', views.chartjs_visualization, name='chartjs_visualization_no_file'),
    path('prediction/<int:file_id>/', views.view_ml_prediction, name='view_ml_prediction'),

    path('portfolio/', views.portfolio, name='portfolio'),
    path('edit-profile/', views.edit_profile, name='edit_profile'),
    

    path('password_reset/', auth_views.PasswordResetView.as_view(template_name='password_reset_form.html'), name='password_reset'),     
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(template_name='password_reset_done.html'), name='password_reset_done'),     
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='password_reset_confirm.html'), name='password_reset_confirm'),     
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name='password_reset_complete.html'), name='password_reset_complete'),
]
 
