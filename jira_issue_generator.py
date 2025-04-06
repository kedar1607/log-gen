#!/usr/bin/env python3
"""
AutoRCA Jira Issue Generator

This script generates realistic Jira issues based on actual log data.
The generated issues include Root Cause Analysis (RCA) information for 80% of the cases
and no RCA for 20% of cases, making it suitable for machine learning model training.

Author: Kedar Haldankar
"""

import argparse
import csv
import datetime
import json
import logging
import os
import random
import sys
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

try:
    from faker import Faker
    import pandas as pd
except ImportError:
    print("Required packages not found. Please install using:")
    print("pip install faker pandas")
    sys.exit(1)

# Configure logging for the script itself
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AutoRCA Jira Generator")

# Initialize Faker for generating realistic data
fake = Faker()

class JiraIssueType(Enum):
    """Enum for Jira issue types"""
    BUG = "Bug"
    INCIDENT = "Incident"
    TASK = "Task"
    STORY = "Story"
    EPIC = "Epic"
    
    @classmethod
    def for_error(cls):
        """Return appropriate issue types for error conditions"""
        return [cls.BUG, cls.INCIDENT]
    
    @classmethod
    def for_performance(cls):
        """Return appropriate issue types for performance issues"""
        return [cls.BUG, cls.TASK]

class JiraPriority(Enum):
    """Enum for Jira priorities"""
    BLOCKER = "Blocker"
    CRITICAL = "Critical"
    MAJOR = "Major"
    MINOR = "Minor"
    TRIVIAL = "Trivial"
    
    @classmethod
    def map_from_log_level(cls, log_level: str) -> "JiraPriority":
        """Maps log level to Jira priority"""
        mapping = {
            "fatal": cls.BLOCKER,
            "critical": cls.BLOCKER,
            "error": cls.CRITICAL,
            "warn": cls.MAJOR,
            "warning": cls.MAJOR,
            "info": cls.MINOR,
            "debug": cls.TRIVIAL,
            "trace": cls.TRIVIAL
        }
        return mapping.get(log_level.lower(), cls.MINOR)

class JiraStatus(Enum):
    """Enum for Jira statuses"""
    OPEN = "Open" 
    IN_PROGRESS = "In Progress"
    CODE_REVIEW = "Code Review"
    TESTING = "Testing"
    DONE = "Done"
    CLOSED = "Closed"
    
    @classmethod
    def random_for_with_rca(cls):
        """Status distribution for issues with RCA"""
        return random.choices(
            [cls.DONE, cls.CLOSED, cls.TESTING, cls.IN_PROGRESS, cls.CODE_REVIEW],
            weights=[0.5, 0.3, 0.1, 0.05, 0.05],
            k=1
        )[0]
    
    @classmethod 
    def random_for_without_rca(cls):
        """Status distribution for issues without RCA"""
        return random.choices(
            [cls.OPEN, cls.IN_PROGRESS, cls.TESTING],
            weights=[0.7, 0.2, 0.1],
            k=1
        )[0]

class JiraRootCause(Enum):
    """Root cause categories"""
    CODE_BUG = "Code Bug"
    PERFORMANCE_BOTTLENECK = "Performance Bottleneck"
    CONFIGURATION_ERROR = "Configuration Error"
    DATABASE_ISSUE = "Database Issue"
    NETWORK_ISSUE = "Network Issue"
    EXTERNAL_DEPENDENCY = "External Dependency Failure"
    RESOURCE_LIMITATION = "Resource Limitation"
    SECURITY_ISSUE = "Security Issue"
    DATA_CORRUPTION = "Data Corruption"
    RACE_CONDITION = "Race Condition"
    MEMORY_LEAK = "Memory Leak"
    
    @classmethod
    def for_service(cls, service: str, error_type: str = None) -> "JiraRootCause":
        """Get likely root causes based on service and error type"""
        # Map common service-specific root causes
        service_map = {
            "api-gateway": [cls.NETWORK_ISSUE, cls.PERFORMANCE_BOTTLENECK, cls.CONFIGURATION_ERROR, cls.EXTERNAL_DEPENDENCY],
            "auth-service": [cls.SECURITY_ISSUE, cls.CONFIGURATION_ERROR, cls.EXTERNAL_DEPENDENCY, cls.CODE_BUG],
            "notification-service": [cls.EXTERNAL_DEPENDENCY, cls.NETWORK_ISSUE, cls.RESOURCE_LIMITATION],
            "user-service": [cls.DATABASE_ISSUE, cls.CODE_BUG, cls.DATA_CORRUPTION],
            "search-service": [cls.PERFORMANCE_BOTTLENECK, cls.RESOURCE_LIMITATION, cls.DATABASE_ISSUE],
            "reporting-service": [cls.DATABASE_ISSUE, cls.PERFORMANCE_BOTTLENECK, cls.MEMORY_LEAK],
            "admin-portal": [cls.CODE_BUG, cls.CONFIGURATION_ERROR, cls.SECURITY_ISSUE],
            "mobile-app": [cls.CODE_BUG, cls.NETWORK_ISSUE, cls.MEMORY_LEAK],
            "frontend": [cls.CODE_BUG, cls.PERFORMANCE_BOTTLENECK, cls.MEMORY_LEAK],
            "referral-service": [cls.DATABASE_ISSUE, cls.CODE_BUG, cls.DATA_CORRUPTION]
        }
        
        # Error type specific mappings to override service defaults
        error_map = {
            "RateLimitExceeded": [cls.RESOURCE_LIMITATION, cls.CONFIGURATION_ERROR],
            "GatewayTimeout": [cls.NETWORK_ISSUE, cls.EXTERNAL_DEPENDENCY],
            "ServiceUnavailable": [cls.EXTERNAL_DEPENDENCY, cls.RESOURCE_LIMITATION],
            "AuthenticationFailed": [cls.SECURITY_ISSUE, cls.CONFIGURATION_ERROR],
            "TokenExpired": [cls.CODE_BUG, cls.CONFIGURATION_ERROR],
            "DatabaseConnectionFailed": [cls.DATABASE_ISSUE, cls.NETWORK_ISSUE],
            "QueryTimeout": [cls.PERFORMANCE_BOTTLENECK, cls.DATABASE_ISSUE],
            "DeliveryFailed": [cls.EXTERNAL_DEPENDENCY, cls.NETWORK_ISSUE],
            "PermissionDenied": [cls.SECURITY_ISSUE, cls.CODE_BUG],
            "ValidationError": [cls.CODE_BUG, cls.DATA_CORRUPTION],
            "SearchIndexCorrupted": [cls.DATA_CORRUPTION, cls.RESOURCE_LIMITATION],
            "BulkOperationFailed": [cls.PERFORMANCE_BOTTLENECK, cls.RESOURCE_LIMITATION]
        }
        
        # If we have an error type and it's in our map, prioritize those root causes
        if error_type and error_type in error_map:
            return random.choice(error_map[error_type])
        
        # Otherwise use service-based root causes or a generic fallback
        return random.choice(service_map.get(service, [cls.CODE_BUG, cls.CONFIGURATION_ERROR, cls.EXTERNAL_DEPENDENCY]))

class JiraIssueGenerator:
    """Main class for generating Jira issues from log data"""
    
    def __init__(
        self,
        logs_directory: str = 'logs',
        rca_ratio: float = 0.8,
        output_directory: str = 'jira_issues',
        limit: int = None
    ):
        """
        Initialize the Jira issue generator
        
        Args:
            logs_directory: Directory containing log CSV files
            rca_ratio: Ratio of issues that should have RCA (0.0-1.0)
            output_directory: Directory to save generated Jira issues
            limit: Maximum number of issues to generate
        """
        self.logs_directory = logs_directory
        self.rca_ratio = min(max(rca_ratio, 0.0), 1.0)  # Clamp between 0-1
        self.output_directory = output_directory
        self.limit = limit
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        # Store consolidated logs
        self.datadog_logs = []
        self.backend_logs = []
        self.analytics_logs = []
        
        # Store issue trackers
        self.issue_counter = 0
        self.issues_with_rca = 0
        self.issues_without_rca = 0
        
        logger.info(f"Initialized Jira issue generator with RCA ratio: {self.rca_ratio:.2%}")
        logger.info(f"Output directory: {self.output_directory}")
    
    def load_logs(self):
        """Load log data from CSV files"""
        # Load DataDog logs
        datadog_path = os.path.join(self.logs_directory, 'datadog_logs.csv')
        if os.path.exists(datadog_path):
            try:
                self.datadog_logs = pd.read_csv(datadog_path).to_dict('records')
                logger.info(f"Loaded {len(self.datadog_logs)} DataDog logs")
            except Exception as e:
                logger.error(f"Error loading DataDog logs: {e}")
        
        # Load Backend logs
        backend_path = os.path.join(self.logs_directory, 'backend_logs.csv')
        if os.path.exists(backend_path):
            try:
                self.backend_logs = pd.read_csv(backend_path).to_dict('records')
                logger.info(f"Loaded {len(self.backend_logs)} Backend logs")
            except Exception as e:
                logger.error(f"Error loading Backend logs: {e}")
        
        # Load Analytics logs (not used for issue generation but might be referenced)
        analytics_path = os.path.join(self.logs_directory, 'analytics_logs.csv')
        if os.path.exists(analytics_path):
            try:
                self.analytics_logs = pd.read_csv(analytics_path).to_dict('records')
                logger.info(f"Loaded {len(self.analytics_logs)} Analytics logs")
            except Exception as e:
                logger.error(f"Error loading Analytics logs: {e}")
    
    def _generate_issue_key(self) -> str:
        """Generate a realistic Jira issue key"""
        project_keys = ["ARCA", "RCA", "SVCOPS", "INFRA", "DEVOPS", "PERF"]
        project = random.choice(project_keys)
        self.issue_counter += 1
        return f"{project}-{random.randint(1000, 9999)}"
    
    def _generate_issue_summary(self, service: str, error_type: str = None, component: str = None) -> str:
        """Generate a realistic issue summary"""
        if error_type:
            templates = [
                f"{error_type} in {service}",
                f"{service}: {error_type} causing service disruption",
                f"{error_type} detected in {service} service",
                f"{service} experiencing {error_type}"
            ]
            if component:
                templates.extend([
                    f"{component} in {service} failing with {error_type}",
                    f"{error_type} in {service}/{component}"
                ])
            return random.choice(templates)
        else:
            # Performance or generic issues
            templates = [
                f"Slow response times in {service}",
                f"{service} performance degradation",
                f"High latency in {service} service",
                f"{service} intermittent timeouts"
            ]
            if component:
                templates.extend([
                    f"{component} in {service} showing degraded performance",
                    f"Slow {component} operations in {service}"
                ])
            return random.choice(templates)
    
    def _generate_assignee(self) -> str:
        """Generate a realistic assignee name"""
        return fake.name()
    
    def _generate_reporter(self) -> str:
        """Generate a realistic reporter name"""
        return fake.name()
    
    def _generate_creation_date(self, log_timestamp: str) -> str:
        """Generate a creation date based on log timestamp"""
        # Parse the original timestamp 
        try:
            log_time = datetime.datetime.fromisoformat(log_timestamp)
            
            # Add a small random delay (incidents typically reported after they occur)
            delay = datetime.timedelta(minutes=random.randint(5, 120))
            creation_time = log_time + delay
            
            return creation_time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            # Fallback to current time if log timestamp can't be parsed
            return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _generate_resolution_date(self, creation_date: str, has_rca: bool) -> Optional[str]:
        """Generate a resolution date based on creation date and RCA status"""
        # Issues without RCA typically don't have resolution dates
        if not has_rca:
            return None
            
        try:
            create_time = datetime.datetime.strptime(creation_date, "%Y-%m-%d %H:%M:%S")
            
            # Add a resolution time (from 2 hours to 7 days)
            resolution_delay = datetime.timedelta(hours=random.randint(2, 24 * 7))
            resolution_time = create_time + resolution_delay
            
            return resolution_time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            # Fallback
            return (datetime.datetime.now() + 
                   datetime.timedelta(days=random.randint(1, 5))).strftime("%Y-%m-%d %H:%M:%S")
    
    def _generate_related_logs(self, service: str, error_type: str = None, timestamp: str = None) -> List[Dict]:
        """Find related logs for a specific service and error type"""
        related_logs = []
        
        # Find datadog logs for this service and issue
        for log in self.datadog_logs[:100]:  # Limit search to avoid performance issues
            if log.get('service') == service:
                if error_type and log.get('error') == error_type:
                    related_logs.append(log)
                elif not error_type and log.get('level') in ['error', 'critical', 'fatal']:
                    related_logs.append(log)
        
        # Find backend logs for this service and issue
        for log in self.backend_logs[:100]:  # Limit search to avoid performance issues
            if log.get('service') == service:
                if error_type and log.get('error_details_type') == error_type:
                    related_logs.append(log)
                elif not error_type and log.get('level') in ['error', 'critical', 'fatal']:
                    related_logs.append(log)
        
        # If we have a timestamp, try to find logs around that time
        if timestamp and not related_logs:
            try:
                log_time = datetime.datetime.fromisoformat(timestamp)
                window_start = (log_time - datetime.timedelta(minutes=15)).isoformat()
                window_end = (log_time + datetime.timedelta(minutes=15)).isoformat()
                
                for log in self.datadog_logs[:100]:
                    if (log.get('service') == service and
                            window_start <= log.get('timestamp', '') <= window_end):
                        related_logs.append(log)
                
                for log in self.backend_logs[:100]:
                    if (log.get('service') == service and
                            window_start <= log.get('timestamp', '') <= window_end):
                        related_logs.append(log)
            except Exception:
                pass
        
        # Return a limited number of logs to avoid excessive data
        return related_logs[:5]
    
    def _generate_rca(self, service: str, error_type: str = None, related_logs: List[Dict] = None) -> Dict:
        """Generate Root Cause Analysis information"""
        root_cause = JiraRootCause.for_service(service, error_type)
        
        # Framework for RCA structure
        rca = {
            "root_cause_category": root_cause.value,
            "affected_components": [service],
            "resolution_steps": [],
            "prevention_steps": []
        }
        
        # Add details based on root cause type
        if root_cause == JiraRootCause.CODE_BUG:
            rca["description"] = f"Issue was identified as a code bug in the {service} service."
            rca["resolution_steps"] = [
                f"Fixed conditional logic in {service} handler",
                "Added unit tests to cover edge case",
                "Deployed hotfix to production"
            ]
            rca["prevention_steps"] = [
                "Improve code review process",
                "Add more comprehensive unit tests",
                "Consider static code analysis"
            ]
            
        elif root_cause == JiraRootCause.PERFORMANCE_BOTTLENECK:
            rca["description"] = f"Performance bottleneck identified in {service} service."
            component = "database query" if random.random() < 0.5 else "API endpoint"
            rca["resolution_steps"] = [
                f"Optimized {component} to reduce response time",
                "Added caching layer to improve performance",
                "Implemented query optimization"
            ]
            rca["prevention_steps"] = [
                "Implement performance testing in CI pipeline",
                "Set up alerting on response time degradation",
                "Regular performance reviews"
            ]
            
        elif root_cause == JiraRootCause.CONFIGURATION_ERROR:
            rca["description"] = f"Misconfiguration in {service} service settings."
            rca["resolution_steps"] = [
                "Updated configuration parameters",
                "Corrected environment variables",
                "Redeployed service with proper settings"
            ]
            rca["prevention_steps"] = [
                "Implement configuration validation",
                "Add configuration tests in deployment pipeline",
                "Document configuration requirements"
            ]
            
        elif root_cause == JiraRootCause.DATABASE_ISSUE:
            rca["description"] = f"Database issue affecting {service} service operations."
            rca["resolution_steps"] = [
                "Fixed database connection pooling",
                "Optimized query execution plan",
                "Added index to improve query performance"
            ]
            rca["prevention_steps"] = [
                "Implement database monitoring",
                "Regular database maintenance",
                "Review query execution plans"
            ]
            
        elif root_cause == JiraRootCause.NETWORK_ISSUE:
            rca["description"] = f"Network connectivity issues impacting {service} service."
            rca["resolution_steps"] = [
                "Increased connection timeouts",
                "Implemented retry mechanism",
                "Added circuit breaker pattern"
            ]
            rca["prevention_steps"] = [
                "Improve network monitoring",
                "Implement more robust error handling",
                "Design for network failure resilience"
            ]
            
        elif root_cause == JiraRootCause.EXTERNAL_DEPENDENCY:
            rca["description"] = f"External dependency failure affecting {service} service."
            dependency = "email service" if "email" in str(related_logs) else "payment processor"
            rca["affected_components"].append(dependency)
            rca["resolution_steps"] = [
                f"Coordinated with {dependency} team to resolve issue",
                "Implemented fallback mechanism",
                "Added better error handling"
            ]
            rca["prevention_steps"] = [
                "Develop SLAs with external providers",
                "Implement circuit breakers",
                "Create fallback mechanisms"
            ]
            
        elif root_cause == JiraRootCause.RESOURCE_LIMITATION:
            rca["description"] = f"Resource limitations in {service} service."
            resource = random.choice(["memory", "CPU", "disk I/O", "connection pool"])
            rca["resolution_steps"] = [
                f"Increased {resource} allocation",
                "Optimized resource usage",
                "Implemented backpressure mechanism"
            ]
            rca["prevention_steps"] = [
                "Set up resource monitoring",
                "Implement autoscaling",
                "Load testing to determine limits"
            ]
            
        elif root_cause == JiraRootCause.SECURITY_ISSUE:
            rca["description"] = f"Security vulnerability in {service} service."
            rca["resolution_steps"] = [
                "Patched security vulnerability",
                "Updated authentication mechanism",
                "Implemented additional validation"
            ]
            rca["prevention_steps"] = [
                "Regular security audits",
                "Implement security scanning in CI/CD",
                "Developer security training"
            ]
            
        elif root_cause == JiraRootCause.DATA_CORRUPTION:
            rca["description"] = f"Data corruption detected in {service} service."
            rca["resolution_steps"] = [
                "Restored data from backup",
                "Fixed data validation logic",
                "Added data integrity checks"
            ]
            rca["prevention_steps"] = [
                "Implement data validation",
                "Regular data integrity checks",
                "Improve backup strategy"
            ]
            
        elif root_cause == JiraRootCause.RACE_CONDITION:
            rca["description"] = f"Race condition identified in {service} service concurrent operations."
            rca["resolution_steps"] = [
                "Implemented proper locking mechanism",
                "Refactored code to eliminate race condition",
                "Added transaction isolation"
            ]
            rca["prevention_steps"] = [
                "Code reviews focusing on concurrency",
                "Stress testing with concurrent operations",
                "Concurrency patterns training"
            ]
            
        elif root_cause == JiraRootCause.MEMORY_LEAK:
            rca["description"] = f"Memory leak detected in {service} service."
            rca["resolution_steps"] = [
                "Fixed resource cleanup in service",
                "Added proper object disposal",
                "Optimized memory usage"
            ]
            rca["prevention_steps"] = [
                "Memory profiling in testing",
                "Set up memory usage monitoring",
                "Regular performance reviews"
            ]
        
        return rca
    
    def _generate_issue_description(self, service: str, log_entry: Dict, has_rca: bool, 
                                 error_type: str = None, related_logs: List[Dict] = None) -> str:
        """Generate a detailed issue description"""
        # Start with a general description
        description = f"## Issue Summary\n"
        
        if error_type:
            description += f"An {error_type} error was detected in the {service} service.\n\n"
        else:
            description += f"Performance degradation or intermittent errors detected in the {service} service.\n\n"
        
        # Add impact statement
        impact = random.choice([
            "This issue is affecting user experience.",
            "This issue is causing service disruption.",
            "This issue is resulting in intermittent failures.",
            "This issue is impacting system performance.",
            "This issue is causing data inconsistency."
        ])
        description += f"**Impact**: {impact}\n\n"
        
        # Add environment details
        env = log_entry.get('env', 'production')
        description += f"**Environment**: {env}\n"
        
        # Add sample logs
        description += "\n## Related Logs\n"
        description += "```\n"
        if related_logs and len(related_logs) > 0:
            for i, log in enumerate(related_logs):
                description += f"{json.dumps(log, indent=2)[:200]}...\n"
                if i < len(related_logs) - 1:
                    description += "---\n"
        else:
            description += f"{json.dumps(log_entry, indent=2)[:300]}...\n"
        description += "```\n\n"
        
        # Add steps to reproduce (for issues with RCA)
        if has_rca:
            description += "## Steps to Reproduce\n"
            description += "1. Trigger the specific operation in the " + service + " service\n"
            
            if "endpoint" in log_entry:
                description += f"2. Access the endpoint {log_entry.get('endpoint')}\n"
            elif "operation" in log_entry:
                description += f"2. Perform the {log_entry.get('operation')} operation\n"
            else:
                description += "2. Under conditions of significant system load\n"
                
            description += "3. Observe the error or performance degradation\n\n"
        
        return description
    
    def _generate_jira_issue(self, log_entry: Dict, source: str = "datadog") -> Dict:
        """Generate a Jira issue from a log entry"""
        # Determine if this issue will have RCA based on the configured ratio
        has_rca = random.random() < self.rca_ratio
        
        # Extract service and error information
        if source == "datadog":
            service = log_entry.get('service', 'unknown')
            error_type = log_entry.get('error')
            level = log_entry.get('level', 'info')
            component = None
            timestamp = log_entry.get('timestamp')
        else:  # backend
            service = log_entry.get('service', 'unknown')
            error_type = log_entry.get('error_details_type')
            level = log_entry.get('level', 'info')
            component = log_entry.get('component')
            timestamp = log_entry.get('timestamp')
        
        # Find related logs for context
        related_logs = self._generate_related_logs(service, error_type, timestamp)
        
        # Generate issue fields
        issue_key = self._generate_issue_key()
        summary = self._generate_issue_summary(service, error_type, component)
        priority = JiraPriority.map_from_log_level(level).value
        
        if error_type:
            issue_type = random.choice(JiraIssueType.for_error()).value
        else:
            issue_type = random.choice(JiraIssueType.for_performance()).value
        
        # Set status based on RCA
        status = JiraStatus.random_for_with_rca().value if has_rca else JiraStatus.random_for_without_rca().value
        
        # Generate timestamps
        creation_date = self._generate_creation_date(timestamp)
        resolution_date = self._generate_resolution_date(creation_date, has_rca) if has_rca else None
        
        # Build the issue
        issue = {
            "key": issue_key,
            "summary": summary,
            "description": self._generate_issue_description(service, log_entry, has_rca, error_type, related_logs),
            "issue_type": issue_type,
            "priority": priority,
            "status": status,
            "assignee": self._generate_assignee(),
            "reporter": self._generate_reporter(),
            "components": [service],
            "created_at": creation_date,
            "resolved_at": resolution_date,
            "affected_services": [service],
            "environment": log_entry.get('env', 'production'),
            "has_rca": has_rca
        }
        
        # Add RCA information for resolved issues
        if has_rca:
            issue["rca"] = self._generate_rca(service, error_type, related_logs)
            self.issues_with_rca += 1
        else:
            self.issues_without_rca += 1
        
        return issue
    
    def generate_issues(self):
        """Generate Jira issues from log data"""
        # Make sure logs are loaded
        if not self.datadog_logs and not self.backend_logs:
            self.load_logs()
            if not self.datadog_logs and not self.backend_logs:
                logger.error("No log data available. Please generate logs first.")
                return
        
        # List to hold all generated issues
        all_issues = []
        
        # Generate issues from DataDog logs (focusing on errors)
        datadog_errors = [log for log in self.datadog_logs 
                          if log.get('level') in ['error', 'critical', 'fatal']]
        
        for log in datadog_errors:
            issue = self._generate_jira_issue(log, source="datadog")
            all_issues.append(issue)
            
            # Check limit
            if self.limit and len(all_issues) >= self.limit:
                break
        
        # If we haven't reached limit, add some from backend errors
        if not self.limit or len(all_issues) < self.limit:
            backend_errors = [log for log in self.backend_logs 
                              if log.get('level') in ['error', 'critical', 'fatal']]
            
            # Figure out how many more we need
            remaining = self.limit - len(all_issues) if self.limit else len(backend_errors)
            
            for log in backend_errors[:remaining]:
                issue = self._generate_jira_issue(log, source="backend")
                all_issues.append(issue)
        
        # If we still need more issues to match our RCA ratio, generate from performance data
        target_with_rca = int(self.rca_ratio * (self.limit or len(all_issues)))
        target_without_rca = (self.limit or len(all_issues)) - target_with_rca
        
        # Check if we need to adjust the RCA distribution
        while self.issues_with_rca < target_with_rca or self.issues_without_rca < target_without_rca:
            # Determine if we need to add an issue with or without RCA
            need_rca = self.issues_with_rca < target_with_rca
            
            # Try to find a suitable log
            if need_rca:
                # For RCA, prefer high latency logs which aren't errors
                potential_logs = [log for log in self.datadog_logs 
                                 if log.get('latency_ms', 0) > 1000 and 
                                 log.get('level') not in ['error', 'critical', 'fatal']]
                if not potential_logs:
                    potential_logs = self.datadog_logs
                
                if potential_logs:
                    log = random.choice(potential_logs)
                    issue = self._generate_jira_issue(log, source="datadog")
                    # Force RCA
                    issue["has_rca"] = True
                    issue["rca"] = self._generate_rca(log.get('service', 'unknown'))
                    all_issues.append(issue)
                    self.issues_with_rca += 1
            else:
                # For non-RCA, take any log and strip RCA
                if self.datadog_logs:
                    log = random.choice(self.datadog_logs)
                    issue = self._generate_jira_issue(log, source="datadog")
                    # Force no RCA
                    issue["has_rca"] = False
                    if "rca" in issue:
                        del issue["rca"]
                    all_issues.append(issue)
                    self.issues_without_rca += 1
            
            # Break if we've reached our limit
            if self.limit and len(all_issues) >= self.limit:
                break
                
            # Also break if we're not making progress
            if not self.datadog_logs:
                break
        
        # Log the counts
        logger.info(f"Generated {len(all_issues)} Jira issues:")
        logger.info(f"- With RCA: {self.issues_with_rca} ({self.issues_with_rca/len(all_issues):.2%})")
        logger.info(f"- Without RCA: {self.issues_without_rca} ({self.issues_without_rca/len(all_issues):.2%})")
        
        # Save the generated issues
        self.save_issues_csv(all_issues)
        
        return all_issues
    
    def save_issues_csv(self, issues: List[Dict]):
        """Save generated issues to CSV file"""
        output_path = os.path.join(self.output_directory, 'jira_issues.csv')
        
        # Prepare data for CSV export
        csv_data = []
        for issue in issues:
            # Flatten RCA data for CSV
            row = issue.copy()
            if issue.get('rca'):
                for key, value in issue['rca'].items():
                    if isinstance(value, list):
                        row[f'rca_{key}'] = '; '.join(value)
                    else:
                        row[f'rca_{key}'] = value
                del row['rca']
            csv_data.append(row)
        
        # Define CSV fields
        if csv_data:
            fieldnames = list(csv_data[0].keys())
            
            # Write to CSV
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            logger.info(f"Saved {len(csv_data)} Jira issues to {output_path}")
        else:
            logger.warning("No issues to save")
    
    def generate_train_test_split(self, test_size: float = 0.2):
        """
        Split the generated issues into training and testing sets
        
        Args:
            test_size: Proportion of data to use for testing (0.0-1.0)
        """
        output_path = os.path.join(self.output_directory, 'jira_issues.csv')
        
        if not os.path.exists(output_path):
            logger.error("No issues file found. Generate issues first.")
            return
        
        try:
            # Load the issues
            issues_df = pd.read_csv(output_path)
            
            # Shuffle the data
            issues_df = issues_df.sample(frac=1.0, random_state=42)
            
            # Calculate split
            test_count = int(len(issues_df) * test_size)
            train_count = len(issues_df) - test_count
            
            # Create the splits
            train_df = issues_df.iloc[:train_count]
            test_df = issues_df.iloc[train_count:]
            
            # Save splits
            train_path = os.path.join(self.output_directory, 'train_issues.csv')
            test_path = os.path.join(self.output_directory, 'test_issues.csv')
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            logger.info(f"Created train-test split:")
            logger.info(f"- Training set: {len(train_df)} issues ({len(train_df)/len(issues_df):.2%})")
            logger.info(f"- Testing set: {len(test_df)} issues ({len(test_df)/len(issues_df):.2%})")
            
            # Count RCA distribution in each set
            train_rca_count = train_df['has_rca'].sum()
            test_rca_count = test_df['has_rca'].sum()
            
            logger.info(f"Training set RCA distribution:")
            logger.info(f"- With RCA: {train_rca_count} ({train_rca_count/len(train_df):.2%})")
            logger.info(f"- Without RCA: {len(train_df)-train_rca_count} ({(len(train_df)-train_rca_count)/len(train_df):.2%})")
            
            logger.info(f"Testing set RCA distribution:")
            logger.info(f"- With RCA: {test_rca_count} ({test_rca_count/len(test_df):.2%})")
            logger.info(f"- Without RCA: {len(test_df)-test_rca_count} ({(len(test_df)-test_rca_count)/len(test_df):.2%})")
            
        except Exception as e:
            logger.error(f"Error creating train-test split: {e}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate Jira issues with RCA from log data')
    
    parser.add_argument('--logs-dir', type=str, default='logs',
                        help='Directory containing log CSV files')
    
    parser.add_argument('--output-dir', type=str, default='jira_issues',
                        help='Directory to save generated Jira issues')
    
    parser.add_argument('--rca-ratio', type=float, default=0.8,
                        help='Proportion of issues that should have RCA information (default: 0.8)')
    
    parser.add_argument('--count', type=int, default=100,
                        help='Number of Jira issues to generate (default: 100)')
                        
    parser.add_argument('--test-split', type=float, default=0.2,
                        help='Proportion of data to use for testing (default: 0.2)')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create the issue generator
    generator = JiraIssueGenerator(
        logs_directory=args.logs_dir,
        rca_ratio=args.rca_ratio,
        output_directory=args.output_dir,
        limit=args.count
    )
    
    # Generate issues from log data
    generator.generate_issues()
    
    # Create train/test split
    generator.generate_train_test_split(test_size=args.test_split)

if __name__ == "__main__":
    main()