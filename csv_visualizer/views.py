import json
from urllib import request
from django.shortcuts import render, redirect,get_object_or_404
import csv
from django.contrib.auth.views import PasswordResetView
from django.urls import reverse_lazy
from django.core.mail import send_mail
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from django.contrib import messages
from .forms import CSVUploadForm, EditProfileForm
import matplotlib.pyplot as plt
import seaborn as sns
from django.contrib.auth.decorators import login_required
import plotly.graph_objects as go
from django.contrib.auth import authenticate,login
import plotly.express as px
from io import BytesIO
from django.contrib.auth import authenticate, login as auth_login
from django.core.files.storage import FileSystemStorage
import pandas as pd
import warnings
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.http import JsonResponse
from .forms import CustomUserCreationForm
from .models import UserProfile
from django.contrib.auth import logout
from .models import CSVFile, AnalysisResult, MLPrediction
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from django.contrib.auth.models import User
from django.db.models import Q
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression 
import joblib 
import random
import io
import statsmodels.api as sm
import base64, urllib
import os
from django.contrib.auth import authenticate, login as auth_login
from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.conf import settings
from django.core.paginator import Paginator
from django.core.exceptions import ValidationError
import logging
logger = logging.getLogger(__name__)



# Suppress UserWarning from Matplotlib
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")



def home(request):
    return render(request, 'home.html')


def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            # Save the user (UserProfile is created automatically in the form's save method)
            form.save()

            # Display success message and redirect to login page
            messages.success(request, "Account created successfully! Please log in.")
            return redirect('/login/')  # Redirect to login view after successful registration
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field.capitalize()}: {error}")
    else:
        form = CustomUserCreationForm()

    return render(request, 'register.html', {'form': form})


def login_view(request):
    # Check if the request method is POST (form submission)
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        try:
            # Check if the user exists in the database
            user = User.objects.get(username=username)

            # Authenticate the user
            user = authenticate(request, username=username, password=password)

            if user is not None:
                # If authentication is successful, log the user in
                auth_login(request, user)
                messages.success(request, 'Login successful!')
                return redirect('/upload/')  # Redirect to upload page or dashboard

            else:
                # Invalid password for the existing user
                messages.error(request, 'Invalid username or password.')

        except User.DoesNotExist:
            # If the username does not exist in the database
            messages.error(request, 'User not found or invalid credentials.')

    # Handle GET request (page load)
    return render(request, 'login.html')

# Helper function to convert Matplotlib figure to PNG image for display
def fig_to_png(fig, save_to_file=False, file_path=None):
    """Helper function to convert a Matplotlib figure to PNG base64 or save as a file."""
    img = BytesIO()  # Initialize the BytesIO object
    fig.savefig(img, format='png')  # Save the figure to the BytesIO object
    img.seek(0)  # Move the pointer to the beginning of the BytesIO object

    if save_to_file:  # If you want to save the image as a file
        if file_path:
            fig.savefig(file_path, format='png')  # Save the figure to the provided file path
            return file_path  # Return the file path to display in the template
        else:
            # Default file path
            default_path = os.path.join(settings.MEDIA_ROOT, 'saved_charts', 'chart.png')
            fig.savefig(default_path, format='png')
            return default_path  # Return the default path to display in the template

    # If not saving to a file, return base64 encoded image
    encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')  # Decode bytes to string for HTML embedding
    return encoded_img


def upload_csv(request):
    # Check user role
    if request.user.userprofile.role == 'viewer':
        # If the user is a viewer, the upload button will be disabled
        upload_disabled = True
    else:
        # If the user is an analyst, the upload button will be enabled
        upload_disabled = False

    # Fetch files uploaded by the current user
    user_files = CSVFile.objects.all()


    # Paginate the files list (10 files per page)
    paginator = Paginator(user_files, 10)  # 10 files per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # First, check the file content before saving it
                file = request.FILES['file']
                file_path = file.name

                # Read the CSV file content using pandas
                df = pd.read_csv(file)

                # Check if the DataFrame is valid (not empty)
                if df.empty:
                    raise ValidationError("The CSV file is empty or couldn't be read correctly.")

                # Check for missing values
                missing_values_count = df.isnull().sum().sum()
                if missing_values_count > 0:
                    # Optional: Handle missing values gracefully (replace, drop, etc.)
                    # Example: Replacing missing values with 'N/A'
                    df.fillna('N/A', inplace=True)
                    messages.warning(request, f"The CSV file contains {missing_values_count} missing values, which have been replaced with 'N/A'.")

                # Add any other validation checks (e.g., checking numeric values, column names, etc.)

                # If all validations pass, save the file
                file_upload = form.save(commit=False)
                file_upload.uploaded_by = request.user  # Associate the current user
                file_upload.save()

                # Perform analysis after saving the file
                # You can now call the analysis or any other post-save operation
                file_upload.perform_analysis()

                # Redirect to the visualization page after a successful upload
                messages.success(request, "CSV file uploaded successfully!")
                return redirect('csv_visualizer:visualize_csv', file_id=file_upload.id)

            except ValidationError as e:
                # Handle validation errors (like missing values)
                messages.error(request, f"Error: {str(e)}")
            except Exception as e:
                # Handle general errors (e.g., invalid CSV format)
                messages.error(request, f"Error: {str(e)}")

        else:
            # Handle invalid form submission
            messages.error(request, "Invalid file. Please upload a valid CSV file.")
    else:
        form = CSVUploadForm()

    return render(request, 'upload_csv.html', {
        'form': form,
        'upload_disabled': upload_disabled,
        'user_files': user_files,
        'page_obj': page_obj
    })




def delete_csv(request, file_id):
    # Get the file object using the provided file_id
    file = get_object_or_404(CSVFile, id=file_id)

    # Check if the current user is the one who uploaded the file
    if file.uploaded_by == request.user:
        try:
            file.delete()  # Delete the file
            messages.success(request, f"File '{file.file.name}' deleted successfully.")
        except Exception as e:
            messages.error(request, f"Error deleting the file: {e}")
    else:
        messages.error(request, "You do not have permission to delete this file.")

    return redirect('csv_visualizer:upload_csv')

def generate_pdf_report(csv_file, analysis_result, bar_chart, boxplot, scatter_chart, bar_chart_plotly, line_chart, pie_chart, violin_plot, stacked_bar_chart, radar_chart, scatter_3d):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    # Title
    p.setFont("Helvetica-Bold", 16)
    p.drawString(200, 750, f"Analysis Report for {csv_file.file.name}")
    p.setFont("Helvetica", 12)
    p.drawString(50, 730, '-' * 100)
    
    # File details
    p.drawString(50, 710, f"File Name: {csv_file.file.name}")
    p.drawString(50, 690, f"Uploaded At: {csv_file.uploaded_at}")
    
    # Add the summary statistics
    y_position = 670
    for stat, value in analysis_result.summary_stats.items():
        p.drawString(50, y_position, f"{stat}: {value}")
        y_position -= 20
    
    # Add each chart to the PDF
    charts = [bar_chart, boxplot, scatter_chart, bar_chart_plotly, line_chart, pie_chart, violin_plot, stacked_bar_chart, radar_chart, scatter_3d]
    chart_y_position = 400
    for chart in charts:
        p.drawString(50, chart_y_position, f"{chart['title']}")
        p.drawImage(chart['image'], 50, chart_y_position - 100, width=500, height=300)
        chart_y_position -= 350  # Adjust for next chart
    
    # Finalize the PDF
    p.showPage()
    p.save()
    
    buffer.seek(0)
    return buffer

def visualize_csv(request, file_id=None):
    # Get the logged-in user
    user = request.user

    if file_id:
        # Get the specific CSV file by file_id
        try:
            csv_file = CSVFile.objects.get(id=file_id, uploaded_by=user)
        except CSVFile.DoesNotExist:
            messages.error(request, "The specified file does not exist.")
    else:
        # Try to get the most recent CSV file uploaded by the user
        try:
            csv_file = CSVFile.objects.filter(uploaded_by=user).latest('uploaded_at')
        except CSVFile.DoesNotExist:
            messages.error(request, "No uploaded files found. Please upload a file first.")

     # If no CSV file is found, redirect to the upload page
    if not csv_file:
        return redirect('csv_visualizer:upload_csv')
    

    # Check if ML predictions exist for this file
    prediction_entry = MLPrediction.objects.filter(data_upload=csv_file).first()

    if not prediction_entry:
        # Generate predictions if not already available (Mock Logic)
        try:
            predictions = [
                {"row": index, "prediction": "Class A" if index % 2 == 0 else "Class B"}
                for index in range(len(df))
            ]
            # Save predictions to the database
            MLPrediction.objects.create(
                data_upload=csv_file,
                predictions=predictions,
                model_name="MockModel",
                status="success"
            )
            messages.success(request, "ML Predictions generated successfully!")
        except Exception as e:
            messages.error(request, f"Error generating predictions: {e}")
            predictions = None
    else:
        # Use existing predictions
        predictions = prediction_entry.predictions


    # Initialize variables
    csv_data = []
    headers = []
    bar_chart = boxplot = scatter_chart = bar_chart_plotly = line_chart = pie_chart = violin_plot = None
    stacked_bar_chart = radar_chart = scatter_3d = None
    prediction_results = None  # To hold the ML prediction results

    if csv_file:
        # Read the CSV file
        try:
            with open(csv_file.file.path, 'r', encoding='ISO-8859-1') as file:
                reader = csv.reader(file)
                headers = next(reader)
                for row in reader:
                    csv_data.append(row)
        except Exception as e:
            csv_data = None
            print(f"Error reading CSV: {e}")


        # Create a DataFrame from the CSV data
        df = pd.DataFrame(csv_data, columns=headers)

        # --- Matplotlib Bar Chart ---
        fig, ax = plt.subplots()
        df.iloc[:, 0].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Bar Chart (Matplotlib)')
        bar_chart = fig_to_png(fig)

        # --- Seaborn Boxplot ---
        fig, ax = plt.subplots()
        sns.boxplot(data=df, ax=ax)
        ax.set_title('Box Plot (Seaborn)')
        boxplot = fig_to_png(fig)

        # --- Plotly Scatter Plot ---
        scatter_fig = px.scatter(df, x=df.columns[0], y=df.columns[1], title="Scatter Plot (Plotly)")
        scatter_chart = scatter_fig.to_html(full_html=False)

        # --- Plotly Bar Chart ---
        bar_fig = px.bar(df, x=df.columns[0], y=df.columns[1], title="Bar Chart (Plotly)")
        bar_chart_plotly = bar_fig.to_html(full_html=False)

        # --- Line Chart ---
        line_fig = px.line(df, x=df.columns[0], y=df.columns[1], title="Line Chart")
        line_chart = line_fig.to_html(full_html=False)

        # --- Pie Chart ---
        pie_fig = px.pie(df, names=df.columns[0], values=df.columns[1], title="Pie Chart")
        pie_chart = pie_fig.to_html(full_html=False)

        # --- Violin Plot ---
        fig, ax = plt.subplots()
        sns.violinplot(x=df.columns[0], y=df.columns[1], data=df, ax=ax)
        ax.set_title('Violin Plot')
        violin_plot = fig_to_png(fig)

        # --- Stacked Bar Chart ---
        stacked_bar_fig = px.bar(df, x=df.columns[0], y=df.columns[1], color=df.columns[2], title="Stacked Bar Chart")
        stacked_bar_chart = stacked_bar_fig.to_html(full_html=False)

        # --- Radar Chart ---
        radar_fig = go.Figure(data=go.Scatterpolar(
            r=df.iloc[0], theta=df.columns, fill='toself'))
        radar_fig.update_layout(title="Radar Chart")
        radar_chart = radar_fig.to_html(full_html=False)

        # --- 3D Scatter Plot ---
        scatter_3d_fig = go.Figure(data=go.Scatter3d(
            x=df.iloc[:, 0], y=df.iloc[:, 1], z=df.iloc[:, 2], mode='markers'))
        scatter_3d_fig.update_layout(title="3D Scatter Plot")
        scatter_3d = scatter_3d_fig.to_html(full_html=False)

    else:
        # If the user has no CSV files uploaded, we set a message
        message = "No data available for visualization. Please upload a CSV file first."

    # Pass all visualizations to the template
    return render(request, 'visualize_csv.html', {
        'csv_file': csv_file,
        'bar_chart': bar_chart,
        'boxplot': boxplot,
        'scatter_chart': scatter_chart,
        'bar_chart_plotly': bar_chart_plotly,
        'line_chart': line_chart,
        'pie_chart': pie_chart,
        'violin_plot': violin_plot,
        'stacked_bar_chart': stacked_bar_chart,
        'radar_chart': radar_chart,
        'scatter_3d': scatter_3d,
         'headers': headers,
        'prediction_results': predictions,
        
            
    })

def custom_logout(request):
    logout(request)
    return redirect('csv_visualizer:login_view')

@login_required
def user_upload_history(request):
    # Get the current logged-in user
    user = request.user

    # Fetch CSV files uploaded by the user
    user_files = CSVFile.objects.filter(uploaded_by=user)

    # Search logic
    search_query = request.GET.get('search', '')
    if search_query:
        # Filter files based on name or upload date using Q objects
        user_files = user_files.filter(
            Q(file__name__icontains=search_query) |
            Q(uploaded_at__icontains=search_query)
        )

    # Filter by sorting criteria (Date, Name, Type)
    filter_by = request.GET.get('filter', 'date')  # Default filter by 'date'
    if filter_by == 'date':
        user_files = user_files.order_by('uploaded_at')
    elif filter_by == 'name':
        user_files = user_files.order_by('file__name')
    elif filter_by == 'type':
        user_files = user_files.order_by('file__type')

    # Optional: Create analysis results for files if not exist
    for file in user_files:
        # Check if analysis result exists for the file
        if not file.analysisresult_set.exists():
            # Create a dummy analysis result if not exists
            summary_stats = {"mean": 5.0, "median": 3.0}  # Replace with actual logic
            AnalysisResult.objects.create(
                data_upload=file,
                summary_stats=summary_stats
            )

    return render(request, 'history.html', {
        'user_files': user_files,
         'search_query': search_query,  # Pass the search query for pre-filling the search box
        'filter_by': filter_by  # Pass the selected filter criteria
    })


def download_csv_file(request, file_id):
    # Get the CSV file object by its ID
    csv_file = get_object_or_404(CSVFile, id=file_id)
    
    # Create a response with the original CSV file
    response = HttpResponse(csv_file.file, content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{csv_file.file.name}"'

    return response


def download_analysis_pdf(request, analysis_id):
    # Get the analysis result object by its ID
    analysis_result = get_object_or_404(AnalysisResult, id=analysis_id)
    
    # Create an HTTP response with PDF content type
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{analysis_result.data_upload.file.name}_analysis_report.pdf"'
    
    # Create a canvas to write to the PDF file
    p = canvas.Canvas(response, pagesize=letter)
    
    # Write the title of the report
    p.setFont("Helvetica-Bold", 16)
    p.drawString(200, 750, f"Analysis Report for {analysis_result.data_upload.file.name}")
    
    # Write a separator line
    p.setFont("Helvetica", 12)
    p.drawString(50, 730, '-' * 100)
    
    # Write the summary statistics in the PDF
    y_position = 710
    for stat, value in analysis_result.summary_stats.items():
        p.drawString(50, y_position, f"{stat}: {value}")
        y_position -= 20
    
    # Finalize the PDF
    p.showPage()
    p.save()
    
    return response



import pandas as pd
import json
import csv
from django.shortcuts import render
from django.contrib import messages
from .models import CSVFile  # Assuming you have a CSVFile model

import pandas as pd
import json
import csv

def chartjs_visualization(request, file_id=None):
    user = request.user
    csv_file = CSVFile.objects.get(id=file_id, uploaded_by=user)

    labels = []
    bar_data = []
    pie_data = []
    line_data = []
    message = None

    try:
        # Read CSV file
        df = pd.read_csv(csv_file.file.path)

        # Automatically detect the columns for labels and numeric data
        categorical_columns = df.select_dtypes(include=['object']).columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

        # Use first categorical column for labels
        if len(categorical_columns) > 0:
            labels = df[categorical_columns[0]].tolist()

        # Use first numeric column for chart data
        if len(numeric_columns) > 0:
            bar_data = df[numeric_columns[0]].tolist()
            pie_data = bar_data
            line_data = bar_data

        # Debugging output
        print("Labels:", labels)
        print("Bar Data:", bar_data)

    except Exception as e:
        message = f"Error processing CSV: {e}"

    return render(request, 'chartjs_visualization.html', {
        'csv_file': csv_file,
        'labels': json.dumps(labels),  # Convert to JSON for Chart.js
        'bar_data': json.dumps(bar_data),
        'pie_data': json.dumps(pie_data),
        'line_data': json.dumps(line_data),
        'message': message,
    })

import pandas as pd

def parse_csv(file_path):
    try:
        # Load the CSV dynamically
        data = pd.read_csv(file_path)

        # Inspect columns and data types
        print("CSV Columns:", data.columns)
        print("First 5 Rows:\n", data.head())

        # Check if the CSV has at least two columns
        if len(data.columns) < 2:
            raise ValueError("CSV must have at least two columns.")

        # Clean the data: Handle missing values
        # Replace missing numerical data with 0, or use another strategy like filling with the mean
        data_cleaned = data.copy()

        # Check if there are any numeric columns and clean them
        for col in data_cleaned.select_dtypes(include=['float64', 'int64']).columns:
            # Handle missing numeric values by filling with 0 or any other method
            data_cleaned[col] = data_cleaned[col].fillna(0)

        # Handle non-numeric columns (e.g., strings), if needed (e.g., filling missing text with 'N/A')
        for col in data_cleaned.select_dtypes(include=['object']).columns:
            data_cleaned[col] = data_cleaned[col].fillna('N/A')

        # Print out cleaned data sample for debugging
        print("Cleaned Data (First 5 Rows):\n", data_cleaned.head())

        # Return cleaned data
        return data_cleaned

    except ValueError as e:
        # Custom error for CSV format or validation errors
        print(f"ValueError: {e}")
        raise e
    except Exception as e:
        # General error for other issues
        print(f"Error reading CSV: {e}")
        raise ValueError(f"Error reading CSV: {e}")

def validate_csv(data):
    if data.empty:
        raise ValueError("The CSV file is empty.")

    if len(data.columns) < 1:
        raise ValueError("The CSV file has no usable columns.")

    # Check for missing values
    if data.isnull().values.any():
        print("Warning: The CSV contains missing values. These will be handled.")

    # Return cleaned data (e.g., drop missing values)
    return data.dropna()

from django.utils.dateparse import parse_date


def portfolio(request):
    user = request.user

    # Get all CSV files uploaded by the user
    uploaded_files = CSVFile.objects.filter(uploaded_by=user).order_by('-uploaded_at')

    # Initialize empty lists for visualizations
    visualizations = []
    predictions = [] 

    # Fetch the visualizations for each file
    for csv_file in uploaded_files:
        # Get the analysis results (visualizations)
        analysis_result = AnalysisResult.objects.filter(data_upload=csv_file).first()

        if analysis_result:
            visualizations.append({
                'file': csv_file,
                'summary_stats': analysis_result.summary_stats,
            })


        # Get Predictions
        prediction_entry = MLPrediction.objects.filter(data_upload=csv_file).first()

        # Add to predictions list if predictions exist
        if prediction_entry and prediction_entry.status == 'success':  # Ensure only successful predictions
            predictions.append({
                'file': csv_file,
                'predictions': prediction_entry.predictions,
                'model_name': prediction_entry.model_name,
                'status': prediction_entry.status,
            })


    # Pagination for uploaded files (10 per page)
    files_paginator = Paginator(uploaded_files, 3)  # 10 files per page
    files_page_number = request.GET.get('files_page')
    files_page_obj = files_paginator.get_page(files_page_number)

    # Pagination for visualizations (5 per page)
    visualizations_paginator = Paginator(visualizations, 3)  # 5 visualizations per page
    visualizations_page_number = request.GET.get('visualizations_page')
    visualizations_page_obj = visualizations_paginator.get_page(visualizations_page_number)

    # Paginate predictions (e.g., 3 per page)
    predictions_paginator = Paginator(predictions, 2)
    predictions_page_number = request.GET.get('predictions_page')
    predictions_page_obj = predictions_paginator.get_page(predictions_page_number)


    return render(request, 'portfolio.html', {
        'files_page_obj': files_page_obj,  # Paginated uploaded files
        'visualizations_page_obj': visualizations_page_obj,  # Paginated visualizations
        'predictions_page_obj': predictions_page_obj,
    })
    
def edit_profile(request):
        user = request.user
        try:
            user_profile = UserProfile.objects.get(user=user)
        except UserProfile.DoesNotExist:
            user_profile = None

        if request.method == 'POST':
            form = EditProfileForm(request.POST, instance=user, user_profile=user_profile)
            if form.is_valid():
                form.save()
                # Update UserProfile role
                if user_profile:
                    user_profile.role = form.cleaned_data['role']
                    user_profile.save()
                messages.success(request, 'Your profile has been updated successfully.')
                return redirect('csv_visualizer:edit_profile')
            else:
                messages.error(request, 'Please correct the errors below.')
        else:
            form = EditProfileForm(instance=user, user_profile=user_profile)

        return render(request, 'edit_profile.html', {'form': form})


def view_ml_prediction(request, file_id):
    try:
        # Fetch the CSV file
        csv_file = CSVFile.objects.get(id=file_id)
        file_path = csv_file.file.path

        # Load data
        data = pd.read_csv(file_path)
        print("Data Loaded:\n", data.head())

        # Ensure Date column exists and is valid
        date_column = None
        for col in data.columns:
            if 'date' in col.lower():
                date_column = col
                break

        if not date_column:
            raise ValueError("No valid Date column found in the file.")

        # Convert Date column to datetime
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
        data.set_index(date_column, inplace=True)

        # Ensure at least one numeric column exists
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_columns) == 0:
            raise ValueError("No valid numeric column found for prediction.")

        # Use the first numeric column for ARIMA
        numeric_column = numeric_columns[0]
        print("Using Numeric Column:", numeric_column)

        # Train ARIMA model
        model = sm.tsa.ARIMA(data[numeric_column], order=(1, 1, 1))
        model_fit = model.fit()

        # Forecast next 5 periods
        forecast = model_fit.forecast(steps=5).tolist()

        # Create graph
        fig, ax = plt.subplots()
        ax.plot(range(len(forecast)), forecast, marker='o', linestyle='-', color='r', label='Predicted Sales')
        ax.set_title(f"ARIMA Predictions for {numeric_column}")
        ax.set_xlabel("Time Periods")
        ax.set_ylabel(numeric_column)
        ax.legend()

        # Convert plot to Base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Save prediction to the database
        ml_prediction = MLPrediction(
            data_upload=csv_file,
            predictions={'forecast': forecast},  # Save forecast as JSON
            model_name='ARIMA',
            status='success',
            visualization=img_str
        )
        ml_prediction.save()

        # Pass to template
        return render(request, 'show_predictions.html', {
            'file_name': csv_file.file.name,
            'predictions': forecast,
            'img_str': img_str,
        })

    except Exception as e:
        # Log error and show friendly message
        print(f"Error in ML Prediction: {e}")
        return render(request, 'show_predictions.html', {
            'file_name': csv_file.file.name,
            'predictions': "Error: Unable to generate predictions.",
            'img_str': None,
            'error': f"Prediction Error: {str(e)}"
        })


class CustomPasswordResetView(PasswordResetView):
    template_name = '  password_reset_form.html'
    email_template_name = 'password_reset_email.html'
    success_url = reverse_lazy('password_reset_done')
    subject_template_name = 'password_reset_subject.txt'

    # Customizing the email sending logic
    def form_valid(self, form):
        email = form.cleaned_data['email']
        # Check if the email exists in your system
        send_mail(
            'Password Reset Request',
            'You requested a password reset. Click the link below to reset your password.',
            settings.DEFAULT_FROM_EMAIL,
            [email],
            fail_silently=False,
        )
        return super().form_valid(form)