#!/usr/bin/env python3
"""
UniteUs Log Generator

This script generates realistic dummy log data for the UniteUs app across
different log sources: DataDog, Analytics, and Backend. The generated logs
are saved in CSV format for later analysis.

Author: AI Assistant
"""

import argparse
import csv
import datetime
import logging
import os
import random
import sys
import time
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
logger = logging.getLogger("UniteUs Log Generator")

# Initialize Faker for generating realistic data
fake = Faker()

class LogLevel(Enum):
    """Enum for log levels across different systems"""
    # Common log levels
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"
    
    # DataDog specific
    TRACE = "trace"
    
    # Get a random log level with weighted probability
    @classmethod
    def random(cls, error_rate: float = 0.05):
        """
        Returns a random log level based on specified error rate
        
        Args:
            error_rate: Float between 0-1 representing probability of error/critical logs
        
        Returns:
            LogLevel: A random log level
        """
        # Default distribution with configurable error rate
        if random.random() < error_rate:
            return random.choice([cls.ERROR, cls.CRITICAL, cls.FATAL])
        else:
            return random.choice([
                cls.INFO, cls.INFO, cls.INFO, cls.INFO,  # Higher weight to INFO
                cls.DEBUG, cls.DEBUG,  # Medium weight to DEBUG
                cls.WARN, cls.WARNING  # Low weight to WARN/WARNING
            ])

class LogSource(Enum):
    """Enum for log sources"""
    DATADOG = "datadog"
    ANALYTICS = "analytics"
    BACKEND = "backend"
    
    @classmethod
    def all(cls):
        """Returns all log sources"""
        return [cls.DATADOG, cls.ANALYTICS, cls.BACKEND]

class UniteUsServices(Enum):
    """Enum for different services in the UniteUs app"""
    API_GATEWAY = "api-gateway"
    AUTH_SERVICE = "auth-service"
    USER_SERVICE = "user-service"
    NOTIFICATION_SERVICE = "notification-service"
    REFERRAL_SERVICE = "referral-service"
    REPORTING_SERVICE = "reporting-service"
    SEARCH_SERVICE = "search-service"
    FRONTEND = "frontend"
    MOBILE_APP = "mobile-app"
    ADMIN_PORTAL = "admin-portal"
    
    @classmethod
    def random(cls):
        """Returns a random UniteUs service"""
        return random.choice(list(cls))

class UserActions(Enum):
    """Enum for common user actions in the UniteUs app"""
    LOGIN = "login"
    LOGOUT = "logout"
    CREATE_PROFILE = "create_profile"
    UPDATE_PROFILE = "update_profile"
    SEARCH = "search"
    VIEW_REFERRAL = "view_referral"
    CREATE_REFERRAL = "create_referral"
    ACCEPT_REFERRAL = "accept_referral"
    DECLINE_REFERRAL = "decline_referral"
    SEND_MESSAGE = "send_message"
    GENERATE_REPORT = "generate_report"
    EXPORT_DATA = "export_data"
    
    @classmethod
    def random(cls):
        """Returns a random user action"""
        return random.choice(list(cls))

class LogGenerator:
    """Main class for generating log data"""
    
    def __init__(
        self,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        error_rate: float = 0.05,
        volume: str = "medium"
    ):
        """
        Initialize the log generator
        
        Args:
            start_time: Starting timestamp for logs (default: 7 days ago)
            end_time: Ending timestamp for logs (default: now)
            error_rate: Probability of error logs (0.0-1.0)
            volume: Log volume - low, medium, high
        """
        self.end_time = end_time or datetime.datetime.now()
        self.start_time = start_time or (self.end_time - datetime.timedelta(days=7))
        self.error_rate = min(max(error_rate, 0.0), 1.0)  # Clamp between 0-1
        
        # Set volume multiplier
        volume_map = {
            "low": 0.5,
            "medium": 1.0,
            "high": 3.0,
            "very_high": 10.0
        }
        self.volume_multiplier = volume_map.get(volume.lower(), 1.0)
        
        # Track user sessions for realistic user flows
        self.active_users = {}
        self.user_agents = self._generate_user_agents(100)
        
        logger.info(f"Initialized log generator from {self.start_time} to {self.end_time}")
        logger.info(f"Error rate: {self.error_rate:.2%}, Volume: {volume} ({self.volume_multiplier}x)")

    def _generate_user_agents(self, count: int) -> List[str]:
        """Generate a list of realistic user agents"""
        user_agents = []
        for _ in range(count):
            user_agents.append(fake.user_agent())
        return user_agents
    
    def _random_ip(self) -> str:
        """Generate a random IP address"""
        return fake.ipv4()
    
    def _random_timestamp(self) -> datetime.datetime:
        """Generate a random timestamp between start_time and end_time"""
        time_diff = (self.end_time - self.start_time).total_seconds()
        random_seconds = random.uniform(0, time_diff)
        return self.start_time + datetime.timedelta(seconds=random_seconds)
    
    def _generate_user_id(self) -> str:
        """Generate a consistent user ID"""
        return f"user_{fake.uuid4()[:8]}"
    
    def _random_latency(self, is_error: bool = False) -> float:
        """Generate a random latency value in ms"""
        # Errors tend to have higher latency
        if is_error:
            return round(random.uniform(500, 10000), 2)
        else:
            # Most requests are fast, with occasional spikes
            if random.random() < 0.9:
                return round(random.uniform(10, 300), 2)
            else:
                return round(random.uniform(300, 2000), 2)
    
    def _generate_error_details(self, service: UniteUsServices) -> Dict:
        """Generate realistic error details"""
        error_types = {
            UniteUsServices.API_GATEWAY: ["RateLimitExceeded", "GatewayTimeout", "ServiceUnavailable"],
            UniteUsServices.AUTH_SERVICE: ["AuthenticationFailed", "TokenExpired", "InvalidCredentials"],
            UniteUsServices.USER_SERVICE: ["UserNotFound", "ValidationError", "DatabaseConnectionFailed"],
            UniteUsServices.NOTIFICATION_SERVICE: ["DeliveryFailed", "TemplateNotFound", "QuotaExceeded"],
            UniteUsServices.REFERRAL_SERVICE: ["InvalidReferralStatus", "ReferralNotFound", "PermissionDenied"],
            UniteUsServices.REPORTING_SERVICE: ["ReportGenerationFailed", "DataFetchFailed", "InvalidParameters"],
            UniteUsServices.SEARCH_SERVICE: ["SearchIndexCorrupted", "QueryTimeout", "IndexingError"],
            UniteUsServices.FRONTEND: ["RenderingError", "ComponentCrashed", "StateManagementFailure"],
            UniteUsServices.MOBILE_APP: ["NetworkError", "AppCrashed", "CacheCorruption"],
            UniteUsServices.ADMIN_PORTAL: ["PermissionDenied", "BulkOperationFailed", "ConfigurationError"]
        }
        
        error_type = random.choice(error_types.get(service, ["UnknownError", "SystemFailure", "InternalError"]))
        
        # Generate stack trace for some errors
        if random.random() < 0.7:
            stack_trace = (
                f"at {service.value}.{error_type}.handleRequest(/{service.value}/{fake.word()}.js:42:15)\n"
                f"at processTicksAndRejections (internal/process/task_queues.js:95:5)\n"
                f"at async {service.value}.{fake.word()}Controller (/{service.value}/controllers/{fake.word()}.js:123:22)"
            )
        else:
            stack_trace = None
            
        return {
            "error_type": error_type,
            "stack_trace": stack_trace,
            "correlation_id": f"corr-{fake.uuid4()}",
            "request_id": f"req-{fake.uuid4()[:8]}"
        }
    
    def _generate_datadog_log(self, timestamp: datetime.datetime) -> Dict:
        """Generate a single DataDog log entry"""
        service = UniteUsServices.random()
        log_level = LogLevel.random(self.error_rate)
        is_error = log_level in [LogLevel.ERROR, LogLevel.CRITICAL, LogLevel.FATAL]
        
        user_id = None
        if random.random() < 0.8:  # 80% of logs have user context
            user_id = self._generate_user_id()
        
        # Base log structure
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "service": service.value,
            "level": log_level.value,
            "env": random.choice(["production", "staging", "development"]),
            "host": f"{service.value}-{random.randint(1,5)}.uniteusapp.io",
            "trace_id": f"trace-{fake.uuid4()}",
            "span_id": f"span-{fake.uuid4()[:8]}",
            "latency_ms": self._random_latency(is_error),
        }
        
        # Add user context if available
        if user_id:
            log_entry["usr"] = {
                "id": user_id,
                "ip": self._random_ip(),
                "agent": random.choice(self.user_agents)
            }
        
        # Add error details for error logs
        if is_error:
            error_details = self._generate_error_details(service)
            log_entry["error"] = error_details["error_type"]
            log_entry["stack"] = error_details["stack_trace"]
            log_entry["correlation_id"] = error_details["correlation_id"]
            log_entry["request_id"] = error_details["request_id"]
            
            # Generate error message
            if service == UniteUsServices.API_GATEWAY:
                log_entry["message"] = f"Request failed: {error_details['error_type']} - Status 5xx returned to client"
            elif service == UniteUsServices.AUTH_SERVICE:
                log_entry["message"] = f"Authentication error: {error_details['error_type']} for user {user_id if user_id else 'anonymous'}"
            else:
                log_entry["message"] = f"Error in {service.value}: {error_details['error_type']}"
        else:
            # Generate normal operation messages
            if service == UniteUsServices.API_GATEWAY:
                endpoint = random.choice(["/api/users", "/api/referrals", "/api/reports", "/api/notifications", "/api/search"])
                method = random.choice(["GET", "POST", "PUT", "DELETE"])
                status = random.choice([200, 201, 204, 400, 401, 403, 404, 429])
                log_entry["message"] = f"{method} {endpoint} - Status: {status} - {log_entry['latency_ms']}ms"
                log_entry["endpoint"] = endpoint
                log_entry["method"] = method
                log_entry["status"] = status
            elif service == UniteUsServices.AUTH_SERVICE:
                action = random.choice(["login", "logout", "token_refresh", "password_reset"])
                log_entry["message"] = f"Auth {action} {'succeeded' if log_level == LogLevel.INFO else 'attempted'}"
                log_entry["auth_action"] = action
            elif service == UniteUsServices.USER_SERVICE:
                action = random.choice(["profile_view", "profile_update", "user_search", "user_created"])
                log_entry["message"] = f"User service: {action}"
                log_entry["user_action"] = action
            else:
                log_entry["message"] = f"{service.value} operation completed in {log_entry['latency_ms']}ms"
        
        # Add additional context based on service
        if service == UniteUsServices.REFERRAL_SERVICE:
            log_entry["referral_id"] = f"ref-{fake.uuid4()[:8]}"
            log_entry["referral_type"] = random.choice(["medical", "housing", "food", "employment", "legal"])
            
        elif service == UniteUsServices.NOTIFICATION_SERVICE:
            log_entry["notification_type"] = random.choice(["email", "sms", "push", "in_app"])
            log_entry["notification_status"] = random.choice(["queued", "sent", "delivered", "failed", "read"])
        
        return log_entry
    
    def _generate_analytics_log(self, timestamp: datetime.datetime) -> Dict:
        """Generate a single Analytics log entry"""
        user_id = self._generate_user_id()
        action = UserActions.random()
        device = random.choice(["desktop", "mobile", "tablet"])
        browser = random.choice(["chrome", "firefox", "safari", "edge"])
        os = random.choice(["windows", "macos", "ios", "android", "linux"])
        
        # Base analytics structure
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "user_id": user_id,
            "session_id": f"sess-{fake.uuid4()[:8]}",
            "event_type": action.value,
            "client_info": {
                "ip": self._random_ip(),
                "user_agent": random.choice(self.user_agents),
                "device": device,
                "browser": browser,
                "os": os,
                "viewport": random.choice(["1920x1080", "1366x768", "375x812", "1440x900"]),
                "language": random.choice(["en-US", "en-GB", "es-ES", "fr-FR"])
            },
            "page": random.choice([
                "/dashboard", 
                "/referrals", 
                "/profile", 
                "/search", 
                "/notifications", 
                "/reports",
                "/admin",
                "/settings",
                "/help"
            ]),
            "referrer": random.choice([
                "", 
                "https://google.com", 
                "https://healthcare.gov", 
                "https://communityresources.org", 
                "https://uniteus.com"
            ])
        }
        
        # Add event-specific details
        if action == UserActions.LOGIN:
            log_entry["login_method"] = random.choice(["password", "sso", "2fa"])
            log_entry["success"] = random.random() > 0.05  # 5% login failure rate
            
        elif action == UserActions.SEARCH:
            log_entry["search_query"] = random.choice([
                "mental health services",
                "food assistance",
                "housing near me",
                "transportation help",
                "childcare resources",
                "elder care"
            ])
            log_entry["search_results_count"] = random.randint(0, 50)
            log_entry["search_filters"] = {
                "distance": random.choice(["5mi", "10mi", "25mi", "50mi"]),
                "service_types": random.sample(["medical", "housing", "food", "transportation", "financial"], k=random.randint(1, 3))
            }
            
        elif action in [UserActions.CREATE_REFERRAL, UserActions.VIEW_REFERRAL, UserActions.ACCEPT_REFERRAL, UserActions.DECLINE_REFERRAL]:
            log_entry["referral_id"] = f"ref-{fake.uuid4()[:8]}"
            log_entry["referral_type"] = random.choice(["medical", "housing", "food", "employment", "legal"])
            log_entry["organization"] = random.choice([
                "County Health Services",
                "Community Housing Alliance",
                "Metro Food Bank",
                "Veterans Support Network",
                "Family Crisis Center"
            ])
            
        elif action == UserActions.GENERATE_REPORT:
            log_entry["report_type"] = random.choice(["utilization", "outcomes", "referrals", "demographics", "network"])
            log_entry["date_range"] = random.choice(["last_week", "last_month", "last_quarter", "ytd", "custom"])
            log_entry["export_format"] = random.choice(["pdf", "csv", "excel"])
            log_entry["generation_time_ms"] = random.randint(500, 15000)
            
        # Add performance metrics
        log_entry["performance"] = {
            "page_load_time_ms": random.randint(100, 5000),
            "first_input_delay_ms": random.randint(10, 500),
            "first_contentful_paint_ms": random.randint(50, 2000)
        }
        
        # Add interaction depth
        log_entry["interaction_depth"] = {
            "scroll_percentage": random.randint(10, 100),
            "time_on_page_seconds": random.randint(5, 1200),
            "clicks": random.randint(0, 20)
        }
        
        return log_entry
    
    def _generate_backend_log(self, timestamp: datetime.datetime) -> Dict:
        """Generate a single Backend log entry"""
        service = UniteUsServices.random()
        log_level = LogLevel.random(self.error_rate)
        is_error = log_level in [LogLevel.ERROR, LogLevel.CRITICAL, LogLevel.FATAL]
        
        # Base backend log structure - more structured than DataDog logs
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "service": service.value,
            "component": random.choice(["api", "worker", "scheduler", "db", "cache"]),
            "level": log_level.value,
            "process_id": random.randint(1000, 9999),
            "thread_id": f"thread-{random.randint(1, 100)}",
            "request_id": f"req-{fake.uuid4()[:8]}",
        }
        
        # Add operation context
        operations = {
            UniteUsServices.API_GATEWAY: ["route_request", "validate_auth", "rate_limit", "transform_response"],
            UniteUsServices.AUTH_SERVICE: ["verify_token", "generate_token", "check_permissions", "update_session"],
            UniteUsServices.USER_SERVICE: ["get_user", "update_user", "create_user", "delete_user", "search_users"],
            UniteUsServices.NOTIFICATION_SERVICE: ["send_notification", "process_template", "check_delivery_status"],
            UniteUsServices.REFERRAL_SERVICE: ["create_referral", "update_status", "assign_referral", "complete_referral"],
            UniteUsServices.REPORTING_SERVICE: ["generate_report", "aggregate_data", "export_report", "schedule_report"],
            UniteUsServices.SEARCH_SERVICE: ["index_document", "search_query", "update_index", "optimize_index"],
            UniteUsServices.FRONTEND: ["render_page", "load_data", "cache_assets", "client_hydration"],
            UniteUsServices.MOBILE_APP: ["sync_data", "background_refresh", "push_registration"],
            UniteUsServices.ADMIN_PORTAL: ["bulk_update", "user_management", "system_configuration", "audit_logs"]
        }
        
        operation = random.choice(operations.get(service, ["unknown_operation"]))
        log_entry["operation"] = operation
        
        # Add database context for some operations
        if "user" in operation or "referral" in operation or random.random() < 0.3:
            log_entry["database"] = {
                "query_type": random.choice(["SELECT", "INSERT", "UPDATE", "DELETE"]),
                "table": random.choice(["users", "referrals", "organizations", "notifications", "sessions"]),
                "execution_time_ms": random.randint(1, 500),
                "rows_affected": random.randint(0, 100)
            }
        
        # Add external API calls for some operations
        if "notification" in operation or "sync" in operation or random.random() < 0.2:
            external_services = ["payment_gateway", "geocoding_api", "email_service", "sms_gateway", "analytics_api"]
            log_entry["external_api"] = {
                "service": random.choice(external_services),
                "endpoint": f"/{fake.word()}/{fake.word()}",
                "status_code": 200 if random.random() > 0.1 else random.choice([400, 401, 403, 429, 500, 503]),
                "response_time_ms": random.randint(50, 2000)
            }
        
        # Add caching info
        if "get" in operation or random.random() < 0.4:
            log_entry["cache"] = {
                "operation": random.choice(["get", "set", "delete", "flush"]),
                "key": f"{service.value}:{fake.word()}:{fake.uuid4()[:8]}",
                "hit": random.random() > 0.3,  # 70% cache hit rate
                "size_bytes": random.randint(10, 50000)
            }
        
        # Generate message based on context
        if is_error:
            error_details = self._generate_error_details(service)
            
            if "database" in log_entry and random.random() < 0.7:
                log_entry["message"] = f"Database error during {operation}: {error_details['error_type']}"
                log_entry["error_code"] = "DB_ERROR"
            elif "external_api" in log_entry and random.random() < 0.7:
                log_entry["message"] = f"External API error: {error_details['error_type']} - {log_entry['external_api']['service']}"
                log_entry["error_code"] = "API_ERROR"
            else:
                log_entry["message"] = f"Error in {service.value}.{operation}: {error_details['error_type']}"
                log_entry["error_code"] = "SYSTEM_ERROR"
                
            log_entry["error_details"] = {
                "type": error_details["error_type"],
                "stack": error_details["stack_trace"],
                "correlation_id": error_details["correlation_id"]
            }
        else:
            # Normal operation messages
            metrics = {}
            
            if "database" in log_entry:
                db_info = log_entry["database"]
                metrics["db_time_ms"] = db_info["execution_time_ms"]
                metrics["rows"] = db_info["rows_affected"]
                
            if "cache" in log_entry:
                cache_info = log_entry["cache"]
                cache_result = "hit" if cache_info["hit"] else "miss"
                metrics["cache"] = cache_result
                
            if "external_api" in log_entry:
                api_info = log_entry["external_api"]
                metrics["api_time_ms"] = api_info["response_time_ms"]
                metrics["status"] = api_info["status_code"]
            
            # Format metrics nicely
            metrics_str = " ".join([f"{k}={v}" for k, v in metrics.items()])
            log_entry["message"] = f"{service.value}.{operation} completed {metrics_str}"
            
            # Add user context when relevant
            if random.random() < 0.6:
                log_entry["user_id"] = self._generate_user_id()
                
            # Add resource utilization for some logs
            if random.random() < 0.3:
                log_entry["resources"] = {
                    "cpu_pct": round(random.uniform(0.1, 95.0), 1),
                    "memory_mb": random.randint(50, 8192),
                    "heap_mb": random.randint(20, 4096)
                }
        
        return log_entry
    
    def generate_logs(self, source: LogSource, count: int) -> List[Dict]:
        """
        Generate logs for a specific source
        
        Args:
            source: The log source to generate logs for
            count: Number of logs to generate
            
        Returns:
            List of log dictionaries
        """
        logs = []
        for _ in range(count):
            timestamp = self._random_timestamp()
            
            if source == LogSource.DATADOG:
                log = self._generate_datadog_log(timestamp)
            elif source == LogSource.ANALYTICS:
                log = self._generate_analytics_log(timestamp)
            elif source == LogSource.BACKEND:
                log = self._generate_backend_log(timestamp)
            else:
                raise ValueError(f"Unknown log source: {source}")
                
            logs.append(log)
            
        # Sort logs by timestamp for realism
        logs.sort(key=lambda x: x["timestamp"])
        return logs
    
    def save_logs_csv(self, logs: List[Dict], filename: str):
        """
        Save logs to a CSV file
        
        Args:
            logs: List of log dictionaries
            filename: Output filename
        """
        if not logs:
            logger.warning(f"No logs to save to {filename}")
            return
            
        # Flatten nested dictionaries for CSV format
        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
            
        flat_logs = [flatten_dict(log) for log in logs]
        
        # Convert to DataFrame for easy CSV export
        df = pd.DataFrame(flat_logs)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(logs)} logs to {filename}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate realistic dummy log data for UniteUs app"
    )
    
    parser.add_argument(
        "--days", 
        type=int, 
        default=7,
        help="Number of days of log data to generate (default: 7)"
    )
    
    parser.add_argument(
        "--error-rate", 
        type=float, 
        default=0.05,
        help="Error rate as a decimal (0.0-1.0, default: 0.05)"
    )
    
    parser.add_argument(
        "--volume", 
        choices=["low", "medium", "high", "very_high"],
        default="medium",
        help="Log volume to generate (default: medium)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./logs",
        help="Directory to save log files (default: ./logs)"
    )
    
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["datadog", "analytics", "backend", "all"],
        default=["all"],
        help="Log sources to generate (default: all)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results (default: random)"
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        fake.seed_instance(args.seed)
        logger.info(f"Using random seed: {args.seed}")
    
    # Calculate date range
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=args.days)
    
    # Initialize log generator
    generator = LogGenerator(
        start_time=start_time,
        end_time=end_time,
        error_rate=args.error_rate,
        volume=args.volume
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which sources to generate
    sources_to_generate = []
    if "all" in args.sources:
        sources_to_generate = LogSource.all()
    else:
        for source_name in args.sources:
            sources_to_generate.append(getattr(LogSource, source_name.upper()))
    
    # Calculate log counts based on volume
    volume_base_counts = {
        LogSource.DATADOG: 5000,
        LogSource.ANALYTICS: 8000,
        LogSource.BACKEND: 10000
    }
    
    # Generate and save logs for each source
    for source in sources_to_generate:
        base_count = volume_base_counts[source]
        count = int(base_count * generator.volume_multiplier * args.days / 7)
        
        logger.info(f"Generating {count} {source.value} logs...")
        logs = generator.generate_logs(source, count)
        
        output_file = os.path.join(args.output_dir, f"{source.value}_logs.csv")
        generator.save_logs_csv(logs, output_file)
    
    logger.info("Log generation complete!")


if __name__ == "__main__":
    main()
