from telegram.ext import ApplicationBuilder
from config import BOT_TOKEN
from bot.handlers import setup_handlers

class BotApp:
    def __init__(self):
        self.application = ApplicationBuilder().token(BOT_TOKEN).build()

    def run(self):
        setup_handlers(self.application)
        print("Бот запущен.")
        self.application.run_polling()