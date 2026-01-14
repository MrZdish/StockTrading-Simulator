# main.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import asyncio
import sys
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')

import nest_asyncio
from telegram.ext import ApplicationBuilder, JobQueue
from utils.logger import setup_logger
from config import BOT_TOKEN
from bot.handlers import setup_handlers

def is_interactive():
    return (
        hasattr(sys, 'ps1') or
        bool(sys.modules.get('spyder')) or
        bool(sys.modules.get('IPython')) or
        'SPYDER' in os.environ or
        'JPY_PARENT_PID' in os.environ
    )

async def main():
    setup_logger()
    
    # Явно создаём JobQueue
    job_queue = JobQueue()
    app = ApplicationBuilder().token(BOT_TOKEN).job_queue(job_queue).build()
    setup_handlers(app)
    print("Бот готов к запуску")
    await app.run_polling()

if __name__ == "__main__":
    
    # Такая манипуляция нужна из за того, что у меня IDE Spyder
    # и для нее нужно вот так делать, чтобы оно запускалось
    if is_interactive():
        print("Обнаружена среда разработки. Применяю nest_asyncio.")
        nest_asyncio.apply()
    asyncio.run(main())