#!/usr/bin/env python3
"""
AutoRCA Log and Jira Issue Generator Web App

This web application provides a user interface for generating log data
and Jira issues with RCA for automatic Root Cause Analysis.

Author: Kedar Haldankar
"""

import os
import json
import datetime
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
import subprocess

# Import the RCA predictor
from rca_predictor import RCAPredictor, perform_rca_prediction, load_jira_issues, save_predicted_issues

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "autorka-secret-key")

# Ensure required directories exist
os.makedirs('logs', exist_ok=True)
os.makedirs('jira_issues', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/generate-logs', methods=['GET', 'POST'])
def generate_logs():
    """Page for generating log data"""
    if request.method == 'POST':
        # Get form parameters
        days = request.form.get('days', '7')
        error_rate = request.form.get('error_rate', '0.05')
        volume = request.form.get('volume', 'medium')
        sources = request.form.getlist('sources') or ['all']
        
        # Build the command
        cmd = ['python', 'log_generator.py', 
               '--days', days,
               '--error-rate', error_rate,
               '--volume', volume]
        
        # Add sources if not 'all'
        if sources != ['all']:
            cmd.extend(['--sources'] + sources)
        
        try:
            # Run the log generator
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            flash(f"Logs generated successfully: {result.stdout}", "success")
        except subprocess.CalledProcessError as e:
            flash(f"Error generating logs: {e.stderr}", "error")
        
        return redirect(url_for('view_logs'))
    
    return render_template('generate_logs.html')

@app.route('/generate-issues', methods=['GET', 'POST'])
def generate_issues():
    """Page for generating Jira issues"""
    if request.method == 'POST':
        # Get form parameters
        count = request.form.get('count', '100')
        rca_ratio = request.form.get('rca_ratio', '0.8')
        test_split = request.form.get('test_split', '0.2')
        
        # Build the command
        cmd = ['python', 'jira_issue_generator.py', 
               '--count', count,
               '--rca-ratio', rca_ratio,
               '--test-split', test_split]
        
        try:
            # Run the Jira issue generator
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            flash(f"Jira issues generated successfully!", "success")
        except subprocess.CalledProcessError as e:
            flash(f"Error generating Jira issues: {e.stderr}", "error")
        
        return redirect(url_for('view_issues'))
    
    return render_template('generate_issues.html')

@app.route('/view-logs')
def view_logs():
    """Page for viewing generated log data"""
    log_files = {}
    
    for source in ['datadog', 'analytics', 'backend']:
        file_path = os.path.join('logs', f'{source}_logs.csv')
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                
                # Get statistics for each log file
                stats = {
                    'count': len(df),
                    'error_count': len(df[df['level'].isin(['error', 'critical', 'fatal'])]) if 'level' in df.columns else 0,
                    'services': df['service'].unique().tolist() if 'service' in df.columns else [],
                    'columns': df.columns.tolist(),
                    'sample': df.head(5).to_dict('records')
                }
                
                log_files[source] = stats
            except Exception as e:
                log_files[source] = {'error': str(e)}
    
    return render_template('view_logs.html', log_files=log_files)

@app.route('/view-issues')
def view_issues():
    """Page for viewing generated Jira issues"""
    issue_files = {}
    
    for file_type in ['jira_issues', 'train_issues', 'test_issues']:
        file_path = os.path.join('jira_issues', f'{file_type}.csv')
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                
                # Get statistics for each file
                stats = {
                    'count': len(df),
                    'with_rca': len(df[df['has_rca'] == True]) if 'has_rca' in df.columns else 0,
                    'without_rca': len(df[df['has_rca'] == False]) if 'has_rca' in df.columns else 0,
                    'issue_types': df['issue_type'].unique().tolist() if 'issue_type' in df.columns else [],
                    'priorities': df['priority'].unique().tolist() if 'priority' in df.columns else [],
                    'statuses': df['status'].unique().tolist() if 'status' in df.columns else [],
                    'columns': df.columns.tolist(),
                    'sample': df.head(5).to_dict('records')
                }
                
                issue_files[file_type] = stats
            except Exception as e:
                issue_files[file_type] = {'error': str(e)}
    
    return render_template('view_issues.html', issue_files=issue_files)

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download generated files"""
    # Check which directory the file is in
    if filename.startswith('logs/'):
        directory = os.path.dirname(filename)
        file = os.path.basename(filename)
        return send_from_directory(directory, file, as_attachment=True)
    elif filename.startswith('jira_issues/'):
        directory = os.path.dirname(filename)
        file = os.path.basename(filename)
        return send_from_directory(directory, file, as_attachment=True)
    else:
        flash("Invalid file path", "error")
        return redirect(url_for('index'))

@app.route('/api/log-summary')
def log_summary():
    """API endpoint to get log summary statistics"""
    summary = {}
    
    # Count logs by type
    for source in ['datadog', 'analytics', 'backend']:
        file_path = os.path.join('logs', f'{source}_logs.csv')
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                summary[source] = {
                    'total': len(df),
                    'by_level': df.groupby('level').size().to_dict() if 'level' in df.columns else {},
                    'by_service': df.groupby('service').size().to_dict() if 'service' in df.columns else {}
                }
            except Exception as e:
                summary[source] = {'error': str(e)}
    
    return jsonify(summary)

@app.route('/api/jira-summary')
def jira_summary():
    """API endpoint to get Jira issue summary statistics"""
    summary = {}
    
    # Get Jira issue statistics
    jira_path = os.path.join('jira_issues', 'jira_issues.csv')
    if os.path.exists(jira_path):
        try:
            df = pd.read_csv(jira_path)
            summary['total'] = len(df)
            summary['with_rca'] = len(df[df['has_rca'] == True]) if 'has_rca' in df.columns else 0
            summary['without_rca'] = len(df[df['has_rca'] == False]) if 'has_rca' in df.columns else 0
            
            # Group by different dimensions
            if 'issue_type' in df.columns:
                summary['by_issue_type'] = df.groupby('issue_type').size().to_dict()
            
            if 'priority' in df.columns:
                summary['by_priority'] = df.groupby('priority').size().to_dict()
            
            if 'status' in df.columns:
                summary['by_status'] = df.groupby('status').size().to_dict()
            
            if 'rca_root_cause_category' in df.columns:
                rca_df = df[df['has_rca'] == True]
                summary['by_root_cause'] = rca_df.groupby('rca_root_cause_category').size().to_dict()
        except Exception as e:
            summary['error'] = str(e)
    
    return jsonify(summary)

@app.route('/predict-rca', methods=['GET', 'POST'])
def predict_rca():
    """Page for predicting RCA for issues without existing RCA"""
    if request.method == 'POST':
        try:
            # Run the ML prediction
            predicted_issues = perform_rca_prediction()
            
            if predicted_issues is not None:
                total_issues = len(predicted_issues)
                issues_with_rca = sum(predicted_issues['has_rca'] == True)
                issues_predicted = sum(predicted_issues.get('predicted_rca', False))
                
                flash(f"Successfully predicted RCA for {issues_predicted} out of {total_issues - issues_with_rca} issues without RCA!", "success")
                flash(f"Predictions saved to jira_issues/predicted_issues.csv", "info")
            else:
                flash("Error occurred during RCA prediction. Check logs for details.", "error")
            
            return redirect(url_for('view_predicted_issues'))
        except Exception as e:
            flash(f"Error in RCA prediction: {str(e)}", "error")
            return redirect(url_for('view_issues'))
    
    # GET request - render the prediction form
    return render_template('predict_rca.html')

@app.route('/view-predicted-issues')
def view_predicted_issues():
    """Page for viewing predicted RCA results"""
    predicted_file = os.path.join('jira_issues', 'predicted_issues.csv')
    
    if not os.path.exists(predicted_file):
        flash("No predicted issues found. Run the RCA predictor first.", "warning")
        return redirect(url_for('predict_rca'))
    
    try:
        # Load predicted issues
        df = pd.read_csv(predicted_file)
        
        # Basic statistics
        stats = {
            'total': len(df),
            'with_original_rca': sum(df['has_rca'] == True),
            'with_predicted_rca': sum(df.get('predicted_rca', False))
        }
        
        # List of predicted issues with confidence scores
        predicted_issues = []
        for _, row in df.iterrows():
            if row.get('predicted_rca', False):
                issue_data = row.to_dict()
                
                # Extract RCA information
                try:
                    if isinstance(issue_data.get('rca'), str):
                        rca_data = json.loads(issue_data['rca'])
                    else:
                        rca_data = issue_data.get('rca', {})
                        
                    issue_data['rca_category'] = rca_data.get('root_cause_category', 'Unknown')
                    issue_data['confidence'] = rca_data.get('confidence', 0)
                    issue_data['alternative_causes'] = rca_data.get('alternative_causes', [])
                    issue_data['evidence'] = rca_data.get('supporting_evidence', [])
                except Exception as e:
                    issue_data['rca_category'] = 'Error parsing RCA'
                    issue_data['confidence'] = 0
                    issue_data['alternative_causes'] = []
                    issue_data['evidence'] = []
                
                predicted_issues.append(issue_data)
        
        # Sort by confidence score
        predicted_issues = sorted(predicted_issues, key=lambda x: x.get('confidence', 0), reverse=True)
        
        return render_template('view_predicted_issues.html', 
                              stats=stats, 
                              predicted_issues=predicted_issues)
    except Exception as e:
        flash(f"Error loading predicted issues: {str(e)}", "error")
        return redirect(url_for('predict_rca'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)