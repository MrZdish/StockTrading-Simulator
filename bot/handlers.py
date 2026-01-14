import os
import pandas as pd
import numpy as np
import asyncio
import matplotlib.pyplot as plt
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ContextTypes, ConversationHandler, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from services.model_ensemble import TechnicalAnalyzer, ModelEnsemble, EnhancedPortfolio, evaluate_metrics
from utils.logger import log_user_action
from config import (
    OFFSET_MONTHS, SIMULATION_SPEED_SEC, FORECAST_HORIZON,
    COMMISSION_RATE, MIN_COMMISSION, INITIAL_CAPITAL_MIN, INITIAL_CAPITAL_MAX, MODEL_CONFIG
)

(
    WAITING_TICKER,
    WAITING_CAPITAL,
    WAITING_MODE,
    WAITING_SUBMODE,
    FAST_SIMULATION,
    LONG_FORECAST
) = range(6)

def _get_progress_bar(current: int, total: int, length: int = 8) -> str:
    if total <= 0:
        return "▰" * length
    filled = int(round(length * current / total))
    bar = "▰" * filled + "▱" * (length - filled)
    return f"{bar} {current}/{total}"

# Обработка команды Start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    log_user_action(user_id, "/start")
    context.user_data.clear()
    await update.message.reply_text("Введите тикер акции (например, AAPL):")
    return WAITING_TICKER

# Обрабатываем тикер акций, которые вводит пользователь
async def receive_ticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ticker = update.message.text.strip().upper()
    user_id = update.effective_user.id
    try:
        stock = __import__('yfinance').Ticker(ticker)
        hist = stock.history(period="5d")
        if hist.empty:
            raise ValueError
        
        # Получаем стоимость акций за 2 года
        df = stock.history(start=(pd.Timestamp.today() - pd.DateOffset(years=2)).strftime('%Y-%m-%d'))
        if df.empty:
            raise ValueError
        
        # Зададим имена столбцов
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Создадим папку для хранения логов, загрузок стоимостей
        user_dir = os.path.join("user_data", str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        
        # Сохраним файл со стоимостью
        df.to_csv(os.path.join(user_dir, f"{ticker}-DAY.csv"))
        context.user_data['ticker'] = ticker
        log_user_action(user_id, f"Тикер принят: {ticker}")
        
        await update.message.reply_text("Введите сумму для инвестиций ($100–$500,000):")
        return WAITING_CAPITAL
    except Exception:
        await update.message.reply_text("Тикер не корректный, введите правильный:")
        return WAITING_TICKER

# Получаем объем деняг, чтобы определить инвестиционные возможности пользователя
async def receive_capital(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        capital = float(update.message.text.replace(',', '').replace('$', ''))
        if not (INITIAL_CAPITAL_MIN <= capital <= INITIAL_CAPITAL_MAX):
            raise ValueError
    except (ValueError, TypeError):
        await update.message.reply_text("Сумма должна быть от $100 до $500,000. Введите снова:")
        return WAITING_CAPITAL

    context.user_data['capital'] = capital
    msg = (
        "Выберите режим:\n\n"
        "Торговля в реальном времени — суточная торговля с сигналами и кнопками\n"
        "Среднесрочное инвестирование — месячный прогноз стоимости портфеля"
    )
    keyboard = [
        [InlineKeyboardButton("Торговля On Line", callback_data="fast"),
         InlineKeyboardButton("Игвестиции на 1 мес.", callback_data="long")]
    ]
    await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(keyboard))
    return WAITING_MODE

async def mode_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    mode = query.data
    context.user_data['mode'] = mode
    context.user_data['chat_id'] = query.message.chat_id

    try:
        await query.delete_message()
    except:
        pass

    # Режим с месячным прогнозом 
    # Просто по ML моделям получаем прогноз по стоимости
    if mode == "long":
        await context.bot.send_message(chat_id=query.message.chat_id, text="Долгий режим выбран. Генерируем прогноз...")

        user_id = query.from_user.id
        ticker = context.user_data['ticker']
        csv_path = os.path.join("user_data", str(user_id), f"{ticker}-DAY.csv")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        analyzer = TechnicalAnalyzer()
        df = analyzer.add_all_indicators(df)
        df = analyzer.clean_data(df)
        await _run_long_forecast(context, df)
        return LONG_FORECAST
    
    # В случае с имитации торговли онлайн
    else:
        
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="Выберите тип быстрой симуляции:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("Автоматическая торговля", callback_data="auto")],
                [InlineKeyboardButton("Полуручная торговля", callback_data="semi")]
            ])
        )
        return WAITING_SUBMODE

def _generate_long_forecast_plot(full_df, forecast, user_id):
    os.makedirs(f"user_data/{user_id}", exist_ok=True)
    plot_path = f"user_data/{user_id}/long_forecast.png"

    # Берём последние 180 дней истории
    hist_180 = full_df[['Close']].tail(180).copy()
    last_date = hist_180.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=len(forecast),
        freq='D'
    )

    plt.figure(figsize=(12, 6))
    plt.plot(hist_180.index, hist_180['Close'], 'k-', linewidth=2, label='История')
    plt.plot(forecast_dates, forecast, 'r--', linewidth=2, label='Прогноз (30 дней)')
    plt.title("Долгосрочный прогноз цены акции")
    plt.ylabel('Цена ($)')
    plt.xlabel('Дата')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path

async def _run_long_forecast(context, full_df):
    chat_id = context.user_data['chat_id']
    capital = context.user_data['capital']
    user_id = context._user_id

    n = len(full_df)
    train_end = int(0.8 * n)
    val_size = n - train_end
    
    # Обучаем ансамбль моделей для предсказания стоимости акций
    ensemble = ModelEnsemble(MODEL_CONFIG)
    ensemble.train_all(full_df, train_end, val_size)

    forecast = ensemble.predict_horizon(full_df, full_df['Close'].iloc[-1], 30)
    final_price = forecast[-1]
    initial_price = full_df['Close'].iloc[-1]
    projected = capital * (final_price / initial_price)

    # Создаем график
    plot_path = _generate_long_forecast_plot(full_df, forecast, user_id)

    # Отправляем график
    with open(plot_path, 'rb') as photo:
        await context.bot.send_photo(chat_id=chat_id, photo=photo)

    # Итоговое сообщение
    msg = (
        f"Прогноз на следующие 30 дней:\n"
        f"Текущая цена: ${initial_price:.2f}\n"
        f"Прогноз: ${final_price:.2f}\n"
        f"Ваш капитал: ${capital:,.2f} → ${projected:,.2f}"
    )
    await context.bot.send_message(chat_id=chat_id, text=msg)
    await context.bot.send_message(chat_id=chat_id, text="До встречи!")

async def submode_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    submode = query.data
    context.user_data['submode'] = submode
    try:
        await query.delete_message()
    except:
        pass

    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text=f"Режим '{'Авто' if submode == 'auto' else 'Полуручной'}' выбран. Начинаем симуляцию..."
    )

    # Загрузка данных
    user_id = query.from_user.id
    ticker = context.user_data['ticker']
    csv_path = os.path.join("user_data", str(user_id), f"{ticker}-DAY.csv")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    analyzer = TechnicalAnalyzer()
    df = analyzer.add_all_indicators(df)
    df = analyzer.clean_data(df)

    # тут мы из общего массива данных срезаем OFFSET_MONTHS, на которых у нас будет
    # происходить симуляционная торговля
    # этот режим необходим для имитации торговли суточной, чтобы по суточному 
    # изменению цены и по сигналам иметь возможность торговать как бы в реальном времени
    
    cutoff = df.index[-1] - pd.DateOffset(months=OFFSET_MONTHS)
    hist = df[df.index <= cutoff].copy()
    sim = df[df.index > cutoff].copy()

    context.user_data['hist'] = hist
    context.user_data['sim'] = sim
    context.user_data['day_index'] = 0
    portfolio = EnhancedPortfolio(context.user_data['capital'], COMMISSION_RATE, MIN_COMMISSION)
    context.user_data['portfolio'] = portfolio

    await _simulate_next_day(context)
    return FAST_SIMULATION

async def _simulate_next_day(context):
    user_data = context.user_data
    day_idx = user_data['day_index']
    sim = user_data['sim']
    hist = user_data['hist']
    portfolio = user_data['portfolio']
    chat_id = user_data['chat_id']
    user_id = context._user_id
    submode = user_data.get('submode', 'semi')

    if day_idx >= len(sim):
        if portfolio.shares > 0:
            final_price = sim['Close'].iloc[-1]
            portfolio.sell(final_price, sim.index[-1], "final_sale")

        # Статистика по сделкам
        buy_trades = [t for t in portfolio.trade_history if t['type'] == 'BUY']
        all_sell_trades = [t for t in portfolio.trade_history if t['type'] == 'SELL']
        successful_sell_trades = [t for t in all_sell_trades if t.get('pnl', 0) > 0]
        
        total_buy_value = sum(t['total_cost'] for t in buy_trades) if buy_trades else 0.0
        total_sell_value = sum(t['proceeds'] for t in successful_sell_trades) if successful_sell_trades else 0.0
        total_commission = sum(t['commission'] for t in portfolio.trade_history)
        net_profit = (total_sell_value - total_buy_value) - total_commission
        final_capital = portfolio.cash + (portfolio.shares * sim['Close'].iloc[-1] if portfolio.shares > 0 else 0)
        
        last_price = sim['Close'].iloc[-1]
        final_capital = portfolio.cash + (portfolio.shares * last_price)

        msg = (
            f"СИМУЛЯЦИЯ ЗАВЕРШЕНА\n\n"
            f"Сделки:\n"
            f"  Покупок: {len(buy_trades)} на ${total_buy_value:,.2f}\n"
            f"  Успешных продаж: {len(successful_sell_trades)} на ${total_sell_value:,.2f}\n"

            f"Портфель:\n"
            f"  Акций в наличии: {portfolio.shares} шт @ ${last_price:.2f}\n"
            f"  Наличные: ${portfolio.cash:,.2f}\n\n"
            f"Финансы:\n"
            f"  Чистая прибыль: ${net_profit:,.2f}\n"
            f"  Комиссии: ${total_commission:,.2f}\n"
            f"  Итоговый капитал: ${final_capital:,.2f}\n\n"
            f"Метрики:\n"
            f"  Sharpe: {portfolio.get_performance_metrics().get('sharpe_ratio', 0):.2f}\n"
            f"  Max DD: {portfolio.get_performance_metrics().get('max_drawdown_pct', 0):.2f}%"
        )
        await context.bot.send_message(chat_id=chat_id, text=msg)
        return ConversationHandler.END

    current_date = sim.index[day_idx]
    current_row = sim.iloc[day_idx]
    current_price = current_row['Close']
    portfolio.update_portfolio(current_price, current_date)

    total_days = len(sim)
    progress_str = _get_progress_bar(day_idx + 1, total_days)

    if len(hist) >= 100:
        try:
            train_end = int(0.8 * len(hist))
            val_size = len(hist) - train_end
            ensemble = ModelEnsemble(MODEL_CONFIG)
            ensemble.train_all(hist, train_end, val_size)
            forecast = ensemble.predict_horizon(hist, current_price, FORECAST_HORIZON)
            forecast_mean = np.mean(forecast)
        except Exception as e:
            log_user_action(user_id, f"Ошибка обучения: {e}")
            forecast_mean = current_price
    else:
        forecast_mean = current_price

    if forecast_mean > current_price * 1.01:
        signal = "BUY"
    elif forecast_mean < current_price * 0.99:
        signal = "SELL"
    else:
        signal = "HOLD"

    log_user_action(user_id, f"{current_date.date()}: {signal} ({submode})")

    if signal == "HOLD":
        
        if submode != "auto":
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"{current_date.date()}\nЦена: ${current_price:.2f}\n{progress_str}\n— просто держим и сидим ровно"
            )
    
        hist = pd.concat([hist, current_row.to_frame().T])
        user_data['hist'] = hist
        user_data['day_index'] += 1

        if context.job_queue is not None:
            context.job_queue.run_once(
                lambda _: _simulate_next_day(context),
                SIMULATION_SPEED_SEC
            )
        else:
            async def delayed():
                await asyncio.sleep(SIMULATION_SPEED_SEC)
                await _simulate_next_day(context)
            asyncio.create_task(delayed())
        return FAST_SIMULATION

    if submode == "auto":
        
        msg = ""
        plot_sent = False
        if signal == "BUY":
            if portfolio.buy(current_price, current_date):
                last_trade = portfolio.trade_history[-1]
                msg = f"   Куплено {last_trade['shares']} акций по ${last_trade['price']:.2f}\n" \
                      f"   На сумму: ${last_trade['total_cost']:.2f} | Комиссия: ${last_trade['commission']:.2f}\n" \
                      f"   Текущий портфель: {portfolio.shares} шт"
                plot_sent = True
            else:
                msg = ""
            
        elif signal == "SELL":
            if portfolio.sell(current_price, current_date, "auto"):
                last_trade = portfolio.trade_history[-1]
                msg = f"   Продано {last_trade['shares']} акций по ${last_trade['price']:.2f}\n" \
                      f"   Выручка: ${last_trade['proceeds']:.2f} | P/L: ${last_trade.get('pnl', 0):.2f}\n" \
                      f"   Текущий портфель: {portfolio.shares} шт"
                plot_sent = True
            else:
                msg = ""
            
        if msg != "":
            await context.bot.send_message(chat_id=chat_id, text=msg)
            
        hist = pd.concat([hist, current_row.to_frame().T])
        user_data['hist'] = hist
        user_data['day_index'] += 1

        if plot_sent:
            plot_path = _generate_signal_plot(
                hist, current_date, current_price, forecast_mean, signal, portfolio,
                user_id, day_idx, total_days, executed=True
            )
            with open(plot_path, 'rb') as photo:
                await context.bot.send_photo(chat_id=chat_id, photo=photo)

        if context.job_queue is not None:
            context.job_queue.run_once(
                lambda _: _simulate_next_day(context),
                SIMULATION_SPEED_SEC
            )
        else:
            async def delayed():
                await asyncio.sleep(SIMULATION_SPEED_SEC)
                await _simulate_next_day(context)
            asyncio.create_task(delayed())
        return FAST_SIMULATION

    else:
        plot_path = _generate_signal_plot(
            hist, current_date, current_price, forecast_mean, signal, portfolio,
            user_id, day_idx, total_days, executed=False
        )
        with open(plot_path, 'rb') as photo:
            await context.bot.send_photo(chat_id=chat_id, photo=photo)

        buttons = []
        if signal == "BUY":
            buttons.append([InlineKeyboardButton("🟢 Купить", callback_data="buy")])
        elif signal == "SELL":
            buttons.append([InlineKeyboardButton("🔴 Продать", callback_data="sell")])
        buttons.append([
            InlineKeyboardButton("⏭ Пропустить", callback_data="skip"),
            InlineKeyboardButton("🛑 Прервать симуляцию", callback_data="end_sim")
        ])
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"{current_date.date()}\nЦена: ${current_price:.2f}\nСигнал: {signal}\n{progress_str}",
            reply_markup=InlineKeyboardMarkup(buttons)
        )
        return FAST_SIMULATION

def _generate_signal_plot(hist, current_date, current_price, forecast_mean, signal, portfolio, user_id, day_idx, total_days, executed=False):
    os.makedirs(f"user_data/{user_id}", exist_ok=True)
    plot_path = f"user_data/{user_id}/signal_{current_date.strftime('%Y%m%d')}.png"

    window_start = current_date - pd.Timedelta(days=60)
    visible_hist = hist[(hist.index >= window_start) & (hist.index <= current_date)]
    plt.figure(figsize=(10, 6))
    plt.plot(visible_hist.index, visible_hist['Close'], 'k-', linewidth=2, label='Цена')
    plt.axhline(y=forecast_mean, color='red', linestyle='--', linewidth=1.5, label='Прогноз')

    # Все реальные сделки (всегда заполненные)
    sim_trades = [t for t in portfolio.trade_history if t['date'] in visible_hist.index]
    buys = [t for t in sim_trades if t['type'] == 'BUY']
    sells = [t for t in sim_trades if t['type'] == 'SELL']
    if buys:
        plt.scatter([t['date'] for t in buys], [t['price'] for t in buys],
                    marker='^', color='green', s=80, edgecolor='black', label='Покупка')
    if sells:
        plt.scatter([t['date'] for t in sells], [t['price'] for t in sells],
                    marker='v', color='red', s=80, edgecolor='black', label='Продажа')

    # Потенциальный сигнал (только если НЕ исполнен и режим полуручной)
    if not executed and signal in ("BUY", "SELL"):
        color = 'green' if signal == "BUY" else 'red'
        marker = '^' if signal == "BUY" else 'v'
        plt.scatter([current_date], [current_price],
                    marker=marker, color=color, s=80,
                    facecolors='none', edgecolors=color, linewidth=2,
                    label=f'Потенц. {signal}')

    progress_str = _get_progress_bar(day_idx + 1, total_days)
    plt.title(f"Сигнал на {current_date.date()} — {signal}\n{progress_str}")
    plt.ylabel('Цена ($)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# Реация на быстрое действие при симуляции торговли
async def handle_fast_action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action = query.data
    user_data = context.user_data
    chat_id = user_data['chat_id']

    if action == "end_sim":
        portfolio = user_data['portfolio']
        metrics = portfolio.get_performance_metrics()
        msg = (
            f"Симуляция остановлена\n"
            f"Капитал: ${metrics['final_value']:,.2f}\n"
            f"Sharpe: {metrics['sharpe_ratio']:.2f}"
        )
        await context.bot.send_message(chat_id=chat_id, text=msg)
        return ConversationHandler.END

    day_idx = user_data['day_index']
    sim = user_data['sim']
    current_date = sim.index[day_idx]
    current_price = sim.iloc[day_idx]['Close']
    portfolio = user_data['portfolio']

    if action == "buy":
        portfolio.buy(current_price, current_date)
        last_trade = portfolio.trade_history[-1]
        msg = f"🟢 Куплено {last_trade['shares']} акций по ${last_trade['price']:.2f}\n" \
              f"   На сумму: ${last_trade['total_cost']:.2f} | Комиссия: ${last_trade['commission']:.2f}\n" \
              f"   Текущий портфель: {portfolio.shares} шт"
        await context.bot.send_message(chat_id=chat_id, text=msg)
        
    elif action == "sell":
        portfolio.sell(current_price, current_date, "manual")
        last_trade = portfolio.trade_history[-1]
        msg = f"🔴 Продано {last_trade['shares']} акций по ${last_trade['price']:.2f}\n" \
              f"   Выручка: ${last_trade['proceeds']:.2f} | P/L: ${last_trade.get('pnl', 0):.2f}\n" \
              f"   Текущий портфель: {portfolio.shares} шт"
        await context.bot.send_message(chat_id=chat_id, text=msg)

    hist = user_data['hist']
    current_row = sim.iloc[day_idx]
    hist = pd.concat([hist, current_row.to_frame().T])
    user_data['hist'] = hist
    user_data['day_index'] += 1

    await _simulate_next_day(context)
    return FAST_SIMULATION

async def end_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("До следующей встречи!")
    return ConversationHandler.END

def setup_handlers(app):
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            WAITING_TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_ticker)],
            WAITING_CAPITAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_capital)],
            WAITING_MODE: [CallbackQueryHandler(mode_choice)],
            WAITING_SUBMODE: [CallbackQueryHandler(submode_choice)],  # ← добавили
            FAST_SIMULATION: [CallbackQueryHandler(handle_fast_action)],
        },
        fallbacks=[CommandHandler("end", end_conversation)],
        allow_reentry=True
    )
    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("end", end_conversation))