import os
import logging
import json
import base64
import time
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from PIL import Image
import io
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Configuration from environment variables
BOT_TOKEN = os.getenv('BOT_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')  # Use ID instead of name
WORKSHEET_NAME = os.getenv('WORKSHEET_NAME', 'Sheet1')  # Default to 'Sheet1'

# Initialize OpenAI client
def get_openai_client():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set!")
    return OpenAI(api_key=OPENAI_API_KEY)

def initialize_google_sheet(max_retries=3):
    """Initializes the gspread client and opens the spreadsheet by ID with retry logic."""
    logging.info("Attempting to connect to Google Sheets...")
    
    # Use exponential backoff for API calls
    for attempt in range(max_retries):
        try:
            # Find credentials file
            possible_paths = [
                '/etc/secrets/credentials.json',  # Render secret files path
                'credentials.json',  # Local development
                '/app/credentials.json',  # Absolute path in container
            ]
            
            creds_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    creds_path = path
                    logging.info(f"Found credentials at: {creds_path}")
                    break
            
            if not creds_path:
                raise FileNotFoundError(f"credentials.json not found in any of: {possible_paths}")
            
            if not SPREADSHEET_ID:
                raise ValueError("SPREADSHEET_ID environment variable is not set!")
            
            # 1. Authenticate using the Service Account file
            gc = gspread.service_account(filename=creds_path)
            
            # 2. Open the spreadsheet using the stable SPREADSHEET_ID
            spreadsheet = gc.open_by_key(SPREADSHEET_ID)
            
            # 3. Select the specific worksheet/tab
            worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
            
            # Create headers if sheet is empty
            if not worksheet.get_all_records():
                worksheet.append_row(["Date", "Store/Merchant", "Total Amount", "Currency", "Transaction Type", "Items", "Timestamp"])
            
            logging.info(f"Google Sheets connection successful. Using Sheet ID: {SPREADSHEET_ID}")
            return worksheet
        
        except gspread.exceptions.APIError as e:
            logging.error(f"Google Sheets API Error on attempt {attempt + 1}: {e}")
            if "PERMISSION_DENIED" in str(e):
                logging.error("Permission denied. Please check if the service account has editor access to the spreadsheet.")
                raise
        except Exception as e:
            logging.error(f"Error on attempt {attempt + 1}: {e}")
            
        if attempt < max_retries - 1:
            # Wait before retrying (exponential backoff)
            wait_time = 2 ** attempt
            logging.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            # Failed all attempts
            logging.error("Failed to connect to Google Sheets after multiple retries.")
            raise

def extract_receipt_info_with_openai(image_bytes):
    """Extract receipt information using OpenAI GPT-4 Vision"""
    try:
        # Get OpenAI client
        client = get_openai_client()
        
        # Convert image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this receipt or transaction image and extract the following information in JSON format:
                            {
                                "store": "store or merchant name",
                                "date": "transaction date",
                                "total_amount": "total amount paid",
                                "currency": "currency used",
                                "transaction_type": "payment type e.g., transfer, purchase, etc.",
                                "items": "list of items purchased or description"
                            }
                            
                            Rules:
                            - If information is not available, use "Unknown"
                            - For amounts, extract only the numerical value without symbols
                            - For dates, use YYYY-MM-DD format if possible
                            - For items, provide a concise description
                            - Be accurate with the currency
                            - For Nigerian receipts, currency is usually NGN
                            """
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # Extract JSON from response
        content = response.choices[0].message.content
        logging.info(f"OpenAI Response: {content}")
        
        # Try to parse JSON from the response
        try:
            # Extract JSON if it's wrapped in markdown code blocks
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            
            receipt_data = json.loads(json_str)
            return receipt_data
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract key information manually
            logging.warning("Failed to parse JSON, extracting manually")
            return extract_info_manually(content)
            
    except Exception as e:
        logging.error(f"OpenAI Vision error: {str(e)}")
        # Return the actual error for debugging
        return {"error": f"OpenAI Error: {str(e)}"}

def extract_info_manually(content):
    """Fallback method to extract information from text response"""
    data = {
        "store": "Unknown",
        "date": "Unknown", 
        "total_amount": "Unknown",
        "currency": "Unknown",
        "transaction_type": "Unknown",
        "items": "Unknown"
    }
    
    # Simple pattern matching as fallback
    content_lower = content.lower()
    
    # Look for store/merchant
    if "opay" in content_lower:
        data["store"] = "OPay"
    
    # Look for amount
    import re
    amount_match = re.search(r'(\d+[,.]?\d*\.?\d{0,2})', content)
    if amount_match:
        data["total_amount"] = amount_match.group(1).replace(',', '')
    
    # Look for date
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',
        r'\d{1,2}/\d{1,2}/\d{4}',
        r'\d{1,2}-\d{1,2}-\d{4}',
        r'\w+ \d{1,2}(?:st|nd|rd|th)?,? \d{4}'
    ]
    
    for pattern in date_patterns:
        date_match = re.search(pattern, content, re.IGNORECASE)
        if date_match:
            data["date"] = date_match.group()
            break
    
    # Look for currency
    if "₦" in content or "naira" in content_lower or "ngn" in content_lower:
        data["currency"] = "NGN"
    
    # Look for transaction type
    if "transfer" in content_lower or "sent" in content_lower:
        data["transaction_type"] = "Transfer"
    elif "purchase" in content_lower or "payment" in content_lower:
        data["transaction_type"] = "Purchase"
    
    return data

# Bot command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to Receipt Bot!\n\n"
        "Simply send me a photo of your receipt and I'll extract the information "
        "and save it to your spreadsheet.\n\n"
        "Supported formats:\n"
        "• Store receipts\n"
        "• Food receipts\n" 
        "• Shopping receipts\n"
        "• Payment transfers\n"
        "• Bank transactions\n\n"
        "Just snap a clear photo and send it to me!"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 How to use Receipt Bot:\n\n"
        "1. Take a clear photo of your receipt\n"
        "2. Send the photo to this chat\n"
        "3. I'll extract the details and save them\n\n"
        "Tips for better results:\n"
        "• Ensure good lighting\n"
        "• Keep the receipt flat\n"
        "• Avoid glare and shadows\n"
        "• Capture the entire receipt"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    processing_msg = None
    try:
        # Check if OpenAI API key is available
        if not OPENAI_API_KEY:
            await update.message.reply_text("❌ OpenAI API key is not configured. Please contact the bot administrator.")
            return
        
        # Send processing message
        processing_msg = await update.message.reply_text("🔄 Processing your receipt with AI...")
        
        # Get the photo file
        photo_file = await update.message.photo[-1].get_file()
        
        # Download photo as bytes
        photo_bytes = await photo_file.download_as_bytearray()
        
        # Extract receipt information using OpenAI
        receipt_data = extract_receipt_info_with_openai(photo_bytes)
        
        # Check if there was an OpenAI error
        if "error" in receipt_data:
            error_message = receipt_data["error"]
            await processing_msg.delete()
            
            # Provide specific guidance based on error type
            if "quota" in error_message.lower() or "billing" in error_message.lower() or "429" in error_message:
                response = "❌ OpenAI API Error: You've exceeded your quota or need to add billing.\n\n"
                response += "Please:\n"
                response += "1. Go to platform.openai.com\n"
                response += "2. Check your billing and usage\n"
                response += "3. Add payment method if needed\n"
                response += f"\nFull Error: {error_message}"
            elif "invalid api key" in error_message.lower():
                response = "❌ OpenAI API Error: Invalid API Key\n\n"
                response += "Please check your OPENAI_API_KEY environment variable in Render."
            else:
                response = f"❌ OpenAI Error: {error_message}"
            
            await update.message.reply_text(response)
            return
        
        if not receipt_data:
            await processing_msg.delete()
            await update.message.reply_text("❌ Sorry, I couldn't process that receipt. Please try again.")
            return
        
        # Save to Google Sheets
        try:
            sheet = initialize_google_sheet()
            timestamp = update.message.date.strftime("%Y-%m-%d %H:%M:%S")
            
            row_data = [
                receipt_data.get('date', 'Unknown'),
                receipt_data.get('store', 'Unknown'),
                receipt_data.get('total_amount', 'Unknown'),
                receipt_data.get('currency', 'Unknown'),
                receipt_data.get('transaction_type', 'Unknown'),
                receipt_data.get('items', 'Unknown'),
                timestamp
            ]
            
            sheet.append_row(row_data)
            
            # Prepare success response message
            response = "✅ Receipt processed and saved!\n\n"
            response += f"🏪 Store: {receipt_data.get('store', 'Unknown')}\n"
            response += f"📅 Date: {receipt_data.get('date', 'Unknown')}\n"
            response += f"💰 Total: {receipt_data.get('currency', '')} {receipt_data.get('total_amount', 'Unknown')}\n"
            response += f"💳 Type: {receipt_data.get('transaction_type', 'Unknown')}\n"
            
            items = receipt_data.get('items', 'Unknown')
            if items and items != 'Unknown':
                response += f"🛍️ Items: {items}\n"
            
            await processing_msg.delete()
            await update.message.reply_text(response)
            
        except Exception as sheets_error:
            logging.error(f"Google Sheets error: {sheets_error}")
            # Even if Sheets fails, show the user what was extracted
            response = "📄 Receipt processed successfully!\n\n"
            response += f"🏪 Store: {receipt_data.get('store', 'Unknown')}\n"
            response += f"📅 Date: {receipt_data.get('date', 'Unknown')}\n"
            response += f"💰 Total: {receipt_data.get('currency', '')} {receipt_data.get('total_amount', 'Unknown')}\n"
            response += f"💳 Type: {receipt_data.get('transaction_type', 'Unknown')}\n\n"
            response += f"⚠️ But couldn't save to spreadsheet. Error: {str(sheets_error)}"
            
            await processing_msg.delete()
            await update.message.reply_text(response)
        
    except Exception as e:
        logging.error(f"Error processing receipt: {str(e)}")
        try:
            if processing_msg:
                await processing_msg.delete()
        except:
            pass
        
        error_response = f"❌ Error processing receipt:\n\n{str(e)}\n\n"
        error_response += "Please try again or contact support if this continues."
        await update.message.reply_text(error_response)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "I only process receipt images right now. "
        "Please send a photo of your receipt, or use /help for instructions."
    )

def main():
    # Check if required environment variables are set
    if not BOT_TOKEN:
        logging.error("BOT_TOKEN environment variable is not set!")
        return
    
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY environment variable is not set!")
        logging.error("Please set the OPENAI_API_KEY environment variable in your Render dashboard")
        return
    
    if not SPREADSHEET_ID:
        logging.error("SPREADSHEET_ID environment variable is not set!")
        logging.error("Please set the SPREADSHEET_ID environment variable in your Render dashboard")
        return
    
    # Create application with specific settings to avoid conflicts
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start bot with specific settings
    logging.info("Bot is starting with OpenAI Vision...")
    
    # Use drop_pending_updates to avoid conflicts
    application.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
