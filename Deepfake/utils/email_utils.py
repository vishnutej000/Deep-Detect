import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Email configuration
EMAIL_HOST = os.getenv("EMAIL_HOST", "")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "")
EMAIL_USE_TLS = os.getenv("EMAIL_USE_TLS", "True").lower() == "true"

def send_email(to_email: str, subject: str, body: str, is_html: bool = False) -> bool:
    """
    Send an email
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        body: Email body content
        is_html: Flag indicating if body is HTML
        
    Returns:
        Boolean indicating success
    """
    # Check if email configuration is set
    if not all([EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASSWORD, EMAIL_FROM]):
        print("Email configuration not set. Cannot send email.")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Attach body based on type
        if is_html:
            msg.attach(MIMEText(body, 'html'))
        else:
            msg.attach(MIMEText(body, 'plain'))
        
        # Connect to server and send
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        if EMAIL_USE_TLS:
            server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        return True
    
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

def send_verification_email(to_email: str, verification_url: str) -> bool:
    """
    Send email verification link
    
    Args:
        to_email: Recipient email address
        verification_url: URL for email verification
        
    Returns:
        Boolean indicating success
    """
    subject = "Verify Your Email Address - Is This Fake?"
    
    # HTML body
    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ width: 100%; max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ text-align: center; margin-bottom: 20px; }}
            .logo {{ font-size: 24px; font-weight: bold; color: #2E86C1; }}
            .content {{ margin: 20px 0; }}
            .button {{ display: inline-block; background: #2E86C1; color: white; padding: 12px 24px; 
                      text-decoration: none; border-radius: 4px; font-weight: bold; }}
            .footer {{ margin-top: 30px; font-size: 12px; color: #666; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">Is This Fake?</div>
            </div>
            <div class="content">
                <p>Hello,</p>
                <p>Thank you for signing up with Is This Fake? To complete your registration, 
                   please verify your email address by clicking the button below:</p>
                <p style="text-align: center;">
                    <a href="{verification_url}" class="button">Verify Your Email</a>
                </p>
                <p>If you did not create an account, please ignore this email.</p>
                <p>This link will expire in 24 hours.</p>
            </div>
            <div class="footer">
                <p>Is This Fake? - AI-Powered Deepfake Detection</p>
                <p>&copy; 2025 Deepfake Detector. All rights reserved.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return send_email(to_email, subject, html_body, is_html=True)

def send_password_reset_email(to_email: str, reset_url: str) -> bool:
    """
    Send password reset link
    
    Args:
        to_email: Recipient email address
        reset_url: URL for password reset
        
    Returns:
        Boolean indicating success
    """
    subject = "Password Reset Request - Is This Fake?"
    
    # HTML body
    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ width: 100%; max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ text-align: center; margin-bottom: 20px; }}
            .logo {{ font-size: 24px; font-weight: bold; color: #2E86C1; }}
            .content {{ margin: 20px 0; }}
            .button {{ display: inline-block; background: #2E86C1; color: white; padding: 12px 24px; 
                      text-decoration: none; border-radius: 4px; font-weight: bold; }}
            .footer {{ margin-top: 30px; font-size: 12px; color: #666; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">Is This Fake?</div>
            </div>
            <div class="content">
                <p>Hello,</p>
                <p>We received a request to reset your password. Click on the button below to create a new password:</p>
                <p style="text-align: center;">
                    <a href="{reset_url}" class="button">Reset Password</a>
                </p>
                <p>If you didn't request a password reset, please ignore this email.</p>
                <p>This link will expire in 1 hour.</p>
            </div>
            <div class="footer">
                <p>Is This Fake? - AI-Powered Deepfake Detection</p>
                <p>&copy; 2025 Deepfake Detector. All rights reserved.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return send_email(to_email, subject, html_body, is_html=True)