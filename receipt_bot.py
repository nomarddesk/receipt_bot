import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import pytesseract
from PIL import Image
import io
import re
import gspread
from google.oauth2.service_account import Credentials
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Configuration from environment variables
BOT_TOKEN = os.getenv('BOT_TOKEN')
SPREADSHEET_NAME = os.getenv('SPREADSHEET_NAME', 'Receipt Tracker')

# For Render.com - we'll use the secret file path
def setup_google_sheets():
    try:
        # On Render, the secret file is mounted at /etc/secrets/
        creds_path = '/etc/secrets/credentials.json'
        
        # Fallback for local development
        if not os.path.exists(creds_path):
            creds_path = 'credentials.json'
            
        scope = ['https://www.googleapis.com/auth/spreadsheets']
        creds = Credentials.from_service_account_file(creds_path, scopes=scope)
        client = gspread.authorize(creds)
        sheet = client.open(SPREADSHEET_NAME).sheet1
        
        # Create headers if sheet is empty
        if not sheet.get_all_records():
            sheet.append_row(["Date", "Store", "Total Amount", "Items", "Timestamp"])
        
        return sheet
    except Exception as e:
        logging.error(f"Google Sheets setup error: {e}")
        raise

# Preprocess image for better OCR
def preprocess_image(image):
    try:
        # Convert to numpy array
        img = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply noise reduction
        denoised = cv2.medianBlur(gray, 5)
        
        # Apply thresholding
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return Image.fromarray(thresh)
    except Exception as e:
        logging.error(f"Image preprocessing error: {e}")
        return image  # Return original if processing fails

# Extract receipt information using OCR
def extract_receipt_info(image):
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Perform OCR
        text = pytesseract.image_to_string(processed_image)
        
        # Parse the extracted text
        receipt_data = parse_receipt_text(text)
        
        return receipt_data, text
    except Exception as e:
        logging.error(f"OCR extraction error: {e}")
        return {}, ""

def parse_receipt_text(text):
    data = {
        'date': '',
        'store': '',
        'total': '',
        'items': ''
    }
    
    lines = text.split('\n')
    
    # Look for date patterns
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',
        r'\d{1,2}-\d{1,2}-\d{2,4}',
        r'\d{1,2}\.\d{1,2}\.\d{2,4}'
    ]
    
    # Look for total amount
    total_patterns = [
        r'total.*?(\d+\.\d{2})',
        r'amount.*?(\d+\.\d{2})',
        r'balance.*?(\d+\.\d{2})',
        r'(\d+\.\d{2})\s*(?:total|amount|balance)',
        r'^.*?(\d+\.\d{2})\s*$'
    ]
    
    # Store name (usually at the top)
    if lines:
        data['store'] = lines[0].strip()[:50]  # Limit length
    
    # Extract date
    for line in lines:
        for pattern in date_patterns:
            match = re.search(pattern, line.lower())
            if match:
                data['date'] = match.group()
                break
    
    # Extract total
    for line in lines:
        for pattern in total_patterns:
            match = re.search(pattern, line.lower())
            if match:
                data['total'] = match.group(1)
                break
    
    # Extract items (simplified)
    item_lines = []
    for line in lines:
        if re.search(r'\d+\.\d{2}', line) and not any(keyword in line.lower() for keyword in ['total', 'tax', 'subtotal', 'amount']):
            item_lines.append(line.strip())
    
    data['items'] = '; '.join(item_lines[:5])[:200]  # Limit to 5 items and 200 chars
    
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
        "• Shopping receipts\n\n"
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
    try:
        # Send processing message
        processing_msg = await update.message.reply_text("🔄 Processing your receipt...")
        
        # Get the photo file
        photo_file = await update.message.photo[-1].get_file()
        
        # Download photo
        photo_bytes = await photo_file.download_as_bytearray()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(photo_bytes))
        
        # Extract receipt information
        receipt_data, raw_text = extract_receipt_info(image)
        
        # Save to Google Sheets
        sheet = setup_google_sheets()
        timestamp = update.message.date.strftime("%Y-%m-%d %H:%M:%S")
        
        row_data = [
            receipt_data['date'],
            receipt_data['store'],
            receipt_data['total'],
            receipt_data['items'],
            timestamp
        ]
        
        sheet.append_row(row_data)
        
        # Prepare response message
        response = "✅ Receipt processed and saved!\n\n"
        response += f"🏪 Store: {receipt_data['store'] or 'Not found'}\n"
        response += f"📅 Date: {receipt_data['date'] or 'Not found'}\n"
        response += f"💰 Total: ${receipt_data['total'] or 'Not found'}\n"
        
        if receipt_data['items']:
            response += f"🛍️ Items: {receipt_data['items']}\n"
        
        await processing_msg.delete()
        await update.message.reply_text(response)
        
    except Exception as e:
        logging.error(f"Error processing receipt: {e}")
        await update.message.reply_text("❌ Sorry, I couldn't process that receipt. Please try again with a clearer image.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "I only process receipt images right now. "
        "Please send a photo of your receipt, or use /help for instructions."
    )

def main():
    # Check if BOT_TOKEN is set
    if not BOT_TOKEN:
        logging.error("BOT_TOKEN environment variable is not set!")
        return
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start bot
    logging.info("Bot is starting...")
    application.run_polling()

if __name__ == "__main__":
    main()
