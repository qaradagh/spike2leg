# spike2leg (SP2L)

ربات معامله‌گر MetaTrader 5 بر پایه‌ی استراتژی **Spike-to-Leg (SP2L / PourSamadi)**.

این ریپازیتوری از پوشه‌ی [`code/SP2L`](https://github.com/AlirezaSadabadi/PythonTraderBot/tree/main/code/SP2L)
پروژه‌ی [PythonTraderBot](https://github.com/AlirezaSadabadi/PythonTraderBot)
نوشته‌ی **Alireza Sadabadi** گرفته شده و **با اجازه‌ی خودش** اینجا منتقل شده تا
توسعه‌ی جداگانه روی آن انجام شود. جزئیات مجوز و انتساب در فایل‌های
[`LICENSE`](LICENSE) و [`NOTICE`](NOTICE) آمده است.

---

## ساختار پروژه

| فایل | توضیح |
| --- | --- |
| `SP2L_Bot.py` | نسخه‌ی ساده و آموزشی ربات؛ منطق استراتژی در تابع `Strategy()` |
| `SP2L_Advanced_Bot.py` | نسخه‌ی کامل با فیلترهای EMA، ساختار روند، ADX و سشن نیویورک |
| `Meta.py` | لایه‌ی ارتباط با MetaTrader 5 (گرفتن کندل، ارسال سفارش، مدیریت پوزیشن، TSL) |
| `TelegramBot.py` | ارسال نوتیفیکیشن به تلگرام (توکن باید پر شود) |
| `SP2L.ipynb` | نوت‌بوک بررسی و تست استراتژی ساده |
| `SP2L2_Advanced_Backtest.ipynb` | بک‌تست نسخه‌ی پیشرفته |
| `requirements.txt` | وابستگی‌ها |

---

## پیش‌نیازها

- **ویندوز** — پکیج `MetaTrader5` فقط روی ویندوز کار می‌کند.
- ترمینال MetaTrader 5 نصب و لاگین‌شده، با فعال بودن *Algo Trading*.
- Python 3.12

## نصب

```powershell
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
```

## اجرا

```powershell
python SP2L_Bot.py           # نسخه ساده
python SP2L_Advanced_Bot.py  # نسخه پیشرفته
```

> ⚠️ قبل از اجرا روی حساب واقعی، حتماً روی حساب **دمو** تست بگیرید.

---

## تنظیمات نسخه‌ی پیشرفته

تنظیمات در بخش `SETTINGS` بالای `SP2L_Advanced_Bot.py` قرار دارد:

| پارامتر | پیش‌فرض | توضیح |
| --- | --- | --- |
| `SYMBOL` | `XAUUSD` | نماد معاملاتی |
| `TIMEFRAME` | `M1` | تایم‌فریم |
| `SPIKE_CANDLE_SIZE` | `1.5` | حداقل نسبت اندازه‌ی کندل اسپایک به کندل‌های اطراف |
| `PGAP_POINTS` | `100` | حداقل فاصله‌ی گپ بین دو پا (بر حسب point بروکر) |
| `MAX_SL_DISTANCE_POINTS` | `1000` | حداکثر فاصله‌ی مجاز استاپ‌لاس |
| `TP_R` | `1.0` | نسبت حد سود به ریسک |
| `USE_EMA_FILTER` / `EMA_PERIOD` | `True` / `60` | فیلتر جهت روند با EMA |
| `USE_TREND_FILTER` / `MAX_OPPOSITE_MOVES` | `True` / `1` | فیلتر ساختار روند |
| `USE_RANGE_FILTER` / `MIN_ADX` | `False` / `20` | فیلتر رِنج با ADX |
| `USE_SESSION_FILTER` | `False` | محدود کردن معاملات به سشن نیویورک |
| `USE_SECOND_ENTRY` | `False` | ورود دوم با حجم ضریب‌دار |
| `LOT` / `MAGIC` | `0.01` / `8` | حجم و شماره‌ی جادویی پوزیشن‌ها |

---

## به‌روزرسانی از ریپوی اصلی

اگر خواستی تغییرات جدید نویسنده‌ی اصلی را ببینی:

```bash
git remote add upstream https://github.com/AlirezaSadabadi/PythonTraderBot.git
git fetch upstream main
git diff HEAD upstream/main -- code/SP2L   # مقایسه دستی و برداشتن تغییرات دلخواه
```

---

## بک‌تست آفلاین

پوشه‌ی `backtest/` یک موتور مستقل است که همان منطق `SP2L2_Advanced_Backtest.ipynb`
را اجرا می‌کند، اما به‌جای MetaTrader 5 از فایل CSV می‌خواند — پس روی لینوکس/مک
هم اجرا می‌شود و کل گرید نمادها در چند ثانیه تمام می‌شود.

```bash
pip install pandas numpy

# اجرای همه‌ی کانفیگ‌ها روی همه‌ی نمادها و تایم‌فریم‌ها
python -m backtest.run_grid --data <data_dir> --out results

# حساسیت نتیجه به قیمت فرضی پرشدن سفارش
python -m backtest.fill_sensitivity --data <data_dir> --out results --spread

# تست‌های موتور
python -m backtest.test_engine
```

`<data_dir>` باید ساختار `<symbol>/<feed>, <timeframe>_<hash>.csv` داشته باشد
(خروجی استاندارد TradingView). هر نماد باید یک `SymbolSpec` در
`backtest/config.py` داشته باشد تا آستانه‌های point-محور درست تبدیل شوند.

### مدل ورود

مهم‌ترین پارامتر `entry_mode` است، چون کل نتیجه به آن حساس است:

| مود | قیمت پرشدن | یعنی چه |
| --- | --- | --- |
| `bar_low` | اکسترمم نهایی کندل ورود | همان چیزی که نوت‌بوک اصلی فرض می‌کند؛ تا بسته‌شدن کندل قابل دانستن نیست |
| `fill_fraction` | کسری از راه بین سطح تریگر تا اکسترمم کندل | مدل واقعی؛ `0.0` یعنی دقیقاً روی سطح تریگر و `1.0` معادل `bar_low` |
| `market_close` | کلوز کندل ورود | اگر ربات فقط روی کندل بسته عمل می‌کرد؛ کران بدبینانه |

ربات زنده `Meta.GetRates` را با `fromDate = now + 3h` صدا می‌زند، پس **کندل در حال
تشکیل** را می‌بیند و به‌محض اینکه low جاری از low کندل قبلی رد شود سفارش مارکت
می‌فرستد. یعنی پرشدن واقعی نزدیک سطح تریگر است، نه اکسترمم نهایی کندل.

نتایج و جزئیات این تفاوت در `results/` و در گزارش تحلیل آمده است.

---

## مجوز

Apache License 2.0 — کد اصلی © Alireza Sadabadi. تغییرات این ریپازیتوری
تحت همان مجوز منتشر می‌شود. جزئیات در [`NOTICE`](NOTICE).

## سلب مسئولیت

این کد صرفاً برای اهداف آموزشی و پژوهشی است. معامله در بازارهای مالی
با اهرم، ریسک از دست دادن کل سرمایه را دارد. مسئولیت استفاده با خودتان است.
