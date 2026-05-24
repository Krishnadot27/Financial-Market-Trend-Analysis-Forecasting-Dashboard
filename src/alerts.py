"""
src/alerts.py
──────────────
Daily prediction alerts via WhatsApp (Twilio) and Email (SMTP).

Sends a daily morning summary of:
  - Tomorrow's prediction for each ticker (UP/DOWN + confidence)
  - Market regime (Bull/Bear/Sideways)
  - Top movers to watch
  - Multi-day outlook (3d/5d)

Setup:
  WhatsApp: needs Twilio account (free trial available)
  Email:    needs Gmail App Password

Configuration in config.yaml:
  alerts:
    email:
      enabled: true
      smtp_host: "smtp.gmail.com"
      smtp_port: 587
      sender: "your@gmail.com"
      password: "your_app_password"
      recipients: ["you@gmail.com"]
    whatsapp:
      enabled: true
      account_sid: "ACxxxxxxxx"
      auth_token:  "xxxxxxxx"
      from_number: "whatsapp:+14155238886"
      to_number:   "whatsapp:+91xxxxxxxxxx"
"""

from __future__ import annotations

import smtplib
import os
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import pandas as pd
from loguru import logger


# ══════════════════════════════════════════════════════════════════════════════
# Message builder
# ══════════════════════════════════════════════════════════════════════════════

def build_alert_message(
    predictions:  dict,
    regime_info:  Optional[dict] = None,
    multiday:     Optional[dict] = None,
    format:       str = "text",
) -> str:
    """
    Build formatted alert message.

    Parameters
    ----------
    predictions : {ticker: {'direction': 'UP'/'DOWN', 'score': float, 'price': float}}
    regime_info : output of RegimeDetector.detect()
    multiday    : {ticker: {'3d': 'UP'/'DOWN', '5d': 'UP'/'DOWN'}}
    format      : 'text' for WhatsApp, 'html' for email

    Returns
    -------
    Formatted string ready to send
    """
    now   = datetime.now().strftime("%d %b %Y, %I:%M %p")
    date  = datetime.now().strftime("%d %b %Y")

    # Split predictions
    bullish = {t: v for t, v in predictions.items() if v.get("direction") == "UP"}
    bearish = {t: v for t, v in predictions.items() if v.get("direction") == "DOWN"}

    if format == "html":
        return _build_html(predictions, bullish, bearish, regime_info, multiday, now, date)
    else:
        return _build_text(predictions, bullish, bearish, regime_info, multiday, now, date)


def _build_text(predictions, bullish, bearish, regime_info, multiday, now, date):
    """Plain text format for WhatsApp."""
    lines = []
    lines.append(f"📈 *NSE Alpha — Daily Prediction*")
    lines.append(f"📅 {date}")
    lines.append("─" * 35)

    # Regime
    if regime_info:
        emoji  = regime_info.get("emoji", "")
        regime = regime_info.get("regime", "UNKNOWN")
        conf   = regime_info.get("confidence", 0)
        lines.append(f"\n{emoji} *Market Regime: {regime}* ({conf:.0%} confidence)")
        lines.append(f"_{regime_info.get('description', '')}_")

    # Bullish picks
    if bullish:
        lines.append(f"\n🟢 *BULLISH ({len(bullish)} stocks)*")
        for ticker, info in sorted(bullish.items(),
                                   key=lambda x: abs(x[1].get("score", 0)),
                                   reverse=True)[:10]:
            t     = ticker.replace(".NS", "")
            score = info.get("score", 0)
            price = info.get("price", 0)
            conf  = info.get("confidence", 0)
            m3d   = multiday.get(ticker, {}).get("3d", "?") if multiday else "?"
            m5d   = multiday.get(ticker, {}).get("5d", "?") if multiday else "?"
            lines.append(
                f"  ▲ *{t}* ₹{price:,.0f} | Score: {score:+.3f} | "
                f"3d:{m3d} 5d:{m5d}"
            )

    # Bearish picks
    if bearish:
        lines.append(f"\n🔴 *BEARISH ({len(bearish)} stocks)*")
        for ticker, info in sorted(bearish.items(),
                                   key=lambda x: abs(x[1].get("score", 0)),
                                   reverse=True)[:10]:
            t     = ticker.replace(".NS", "")
            score = info.get("score", 0)
            price = info.get("price", 0)
            m3d   = multiday.get(ticker, {}).get("3d", "?") if multiday else "?"
            m5d   = multiday.get(ticker, {}).get("5d", "?") if multiday else "?"
            lines.append(
                f"  ▼ *{t}* ₹{price:,.0f} | Score: {score:+.3f} | "
                f"3d:{m3d} 5d:{m5d}"
            )

    lines.append("\n─" * 35)
    lines.append(f"⚠️ _Not financial advice. For research only._")
    lines.append(f"🤖 _NSE Alpha ML System | {now}_")

    return "\n".join(lines)


def _build_html(predictions, bullish, bearish, regime_info, multiday, now, date):
    """Rich HTML format for email."""
    regime_color = regime_info.get("color", "#3B82F6") if regime_info else "#3B82F6"
    regime_name  = regime_info.get("regime", "UNKNOWN") if regime_info else "UNKNOWN"
    regime_emoji = regime_info.get("emoji", "") if regime_info else ""
    regime_desc  = regime_info.get("description", "") if regime_info else ""
    regime_conf  = f"{regime_info.get('confidence', 0):.0%}" if regime_info else "—"

    # Build rows
    bull_rows = ""
    for ticker, info in sorted(bullish.items(),
                                key=lambda x: abs(x[1].get("score", 0)),
                                reverse=True):
        t     = ticker.replace(".NS", "")
        score = info.get("score", 0)
        price = info.get("price", 0)
        m3d   = multiday.get(ticker, {}).get("3d", "—") if multiday else "—"
        m5d   = multiday.get(ticker, {}).get("5d", "—") if multiday else "—"
        bull_rows += f"""
        <tr>
          <td style="padding:8px;font-weight:600;color:#F8FAFC">{t}</td>
          <td style="padding:8px;color:#94A3B8">₹{price:,.0f}</td>
          <td style="padding:8px;color:#22C55E;font-weight:600">▲ UP</td>
          <td style="padding:8px;color:#22C55E;font-family:monospace">{score:+.4f}</td>
          <td style="padding:8px;color:#{'22C55E' if m3d=='UP' else 'EF4444' if m3d=='DOWN' else '94A3B8'}">{m3d}</td>
          <td style="padding:8px;color:#{'22C55E' if m5d=='UP' else 'EF4444' if m5d=='DOWN' else '94A3B8'}">{m5d}</td>
        </tr>"""

    bear_rows = ""
    for ticker, info in sorted(bearish.items(),
                                key=lambda x: abs(x[1].get("score", 0)),
                                reverse=True):
        t     = ticker.replace(".NS", "")
        score = info.get("score", 0)
        price = info.get("price", 0)
        m3d   = multiday.get(ticker, {}).get("3d", "—") if multiday else "—"
        m5d   = multiday.get(ticker, {}).get("5d", "—") if multiday else "—"
        bear_rows += f"""
        <tr>
          <td style="padding:8px;font-weight:600;color:#F8FAFC">{t}</td>
          <td style="padding:8px;color:#94A3B8">₹{price:,.0f}</td>
          <td style="padding:8px;color:#EF4444;font-weight:600">▼ DOWN</td>
          <td style="padding:8px;color:#EF4444;font-family:monospace">{score:+.4f}</td>
          <td style="padding:8px;color:#{'22C55E' if m3d=='UP' else 'EF4444' if m3d=='DOWN' else '94A3B8'}">{m3d}</td>
          <td style="padding:8px;color:#{'22C55E' if m5d=='UP' else 'EF4444' if m5d=='DOWN' else '94A3B8'}">{m5d}</td>
        </tr>"""

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="background:#0A0E1A;color:#E2E8F0;font-family:'Segoe UI',sans-serif;margin:0;padding:20px">
  <div style="max-width:700px;margin:0 auto">

    <!-- Header -->
    <div style="background:linear-gradient(135deg,#1E293B,#0F1729);border:1px solid #1E293B;
                border-radius:16px;padding:28px;margin-bottom:20px;text-align:center">
      <div style="font-size:2rem;font-weight:800;letter-spacing:-0.03em;color:#F8FAFC">
        📈 NSE Alpha
      </div>
      <div style="color:#475569;font-size:0.85rem;margin-top:4px">
        Daily ML Prediction Report — {date}
      </div>
    </div>

    <!-- Regime -->
    <div style="background:#111827;border:1px solid {regime_color}44;border-left:4px solid {regime_color};
                border-radius:12px;padding:20px;margin-bottom:20px">
      <div style="font-size:1.3rem;font-weight:700;color:{regime_color}">
        {regime_emoji} Market Regime: {regime_name}
        <span style="font-size:0.9rem;color:#64748B;font-weight:400">({regime_conf})</span>
      </div>
      <div style="color:#94A3B8;font-size:0.85rem;margin-top:6px">{regime_desc}</div>
    </div>

    <!-- Summary -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px">
      <div style="background:#052e16;border:1px solid #22C55E44;border-radius:12px;padding:16px;text-align:center">
        <div style="font-size:2rem;font-weight:800;color:#22C55E">{len(bullish)}</div>
        <div style="color:#86EFAC;font-size:0.8rem">Bullish Signals</div>
      </div>
      <div style="background:#450a0a;border:1px solid #EF444444;border-radius:12px;padding:16px;text-align:center">
        <div style="font-size:2rem;font-weight:800;color:#EF4444">{len(bearish)}</div>
        <div style="color:#FCA5A5;font-size:0.8rem">Bearish Signals</div>
      </div>
    </div>

    <!-- Table -->
    <div style="background:#111827;border:1px solid #1E293B;border-radius:12px;overflow:hidden;margin-bottom:20px">
      <table style="width:100%;border-collapse:collapse">
        <thead>
          <tr style="background:#1E293B">
            <th style="padding:10px 8px;text-align:left;color:#64748B;font-size:0.8rem">TICKER</th>
            <th style="padding:10px 8px;text-align:left;color:#64748B;font-size:0.8rem">PRICE</th>
            <th style="padding:10px 8px;text-align:left;color:#64748B;font-size:0.8rem">SIGNAL</th>
            <th style="padding:10px 8px;text-align:left;color:#64748B;font-size:0.8rem">SCORE</th>
            <th style="padding:10px 8px;text-align:left;color:#64748B;font-size:0.8rem">3-DAY</th>
            <th style="padding:10px 8px;text-align:left;color:#64748B;font-size:0.8rem">5-DAY</th>
          </tr>
        </thead>
        <tbody>
          {bull_rows}
          {bear_rows}
        </tbody>
      </table>
    </div>

    <!-- Footer -->
    <div style="text-align:center;color:#334155;font-size:0.75rem;padding:12px">
      ⚠️ Not financial advice. For research and educational purposes only.<br>
      Generated by NSE Alpha ML System at {now}
    </div>
  </div>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# Email sender
# ══════════════════════════════════════════════════════════════════════════════

class EmailAlerter:
    """
    Send daily prediction alerts via Gmail SMTP.

    Setup:
      1. Enable 2FA on your Gmail account
      2. Go to myaccount.google.com → Security → App Passwords
      3. Generate a password for "Mail"
      4. Use that 16-char password in config.yaml
    """

    def __init__(self, config: dict) -> None:
        cfg              = config.get("alerts", {}).get("email", {})
        self.enabled     = cfg.get("enabled", False)
        self.smtp_host   = cfg.get("smtp_host",  "smtp.gmail.com")
        self.smtp_port   = cfg.get("smtp_port",  587)
        self.sender      = cfg.get("sender",     "")
        self.password    = cfg.get("password",   os.getenv("EMAIL_PASSWORD", ""))
        self.recipients  = cfg.get("recipients", [])

    def send(
        self,
        predictions:  dict,
        regime_info:  Optional[dict] = None,
        multiday:     Optional[dict] = None,
    ) -> bool:
        """
        Send HTML email with predictions.

        Returns True if sent successfully, False otherwise.
        """
        if not self.enabled:
            logger.info("Email alerts disabled in config")
            return False

        if not self.sender or not self.password or not self.recipients:
            logger.warning(
                "Email not configured. Add to config.yaml:\n"
                "  alerts:\n"
                "    email:\n"
                "      enabled: true\n"
                "      sender: your@gmail.com\n"
                "      password: your_app_password\n"
                "      recipients: [you@gmail.com]"
            )
            return False

        try:
            html_body = build_alert_message(
                predictions, regime_info, multiday, format="html"
            )
            date = datetime.now().strftime("%d %b %Y")

            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"📈 NSE Alpha — Daily Predictions {date}"
            msg["From"]    = self.sender
            msg["To"]      = ", ".join(self.recipients)
            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender, self.password)
                server.sendmail(
                    self.sender,
                    self.recipients,
                    msg.as_string(),
                )

            logger.success(f"Email sent to {self.recipients}")
            return True

        except Exception as e:
            logger.error(f"Email failed: {e}")
            return False


# ══════════════════════════════════════════════════════════════════════════════
# WhatsApp sender (Twilio)
# ══════════════════════════════════════════════════════════════════════════════

class WhatsAppAlerter:
    """
    Send daily prediction alerts via WhatsApp using Twilio.

    Setup (free):
      1. Sign up at twilio.com (free trial gives $15 credit)
      2. Enable WhatsApp Sandbox at console.twilio.com/messaging/whatsapp
      3. Send "join <your-sandbox-word>" to +14155238886 from your WhatsApp
      4. Copy Account SID + Auth Token from Twilio Console
      5. Add to config.yaml (see below)

    config.yaml:
      alerts:
        whatsapp:
          enabled: true
          account_sid: "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
          auth_token:  "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
          from_number: "whatsapp:+14155238886"
          to_number:   "whatsapp:+91XXXXXXXXXX"
    """

    def __init__(self, config: dict) -> None:
        cfg               = config.get("alerts", {}).get("whatsapp", {})
        self.enabled      = cfg.get("enabled",     False)
        self.account_sid  = cfg.get("account_sid", os.getenv("TWILIO_SID",   ""))
        self.auth_token   = cfg.get("auth_token",  os.getenv("TWILIO_TOKEN", ""))
        self.from_number  = cfg.get("from_number", "whatsapp:+14155238886")
        self.to_number    = cfg.get("to_number",   "")

    def send(
        self,
        predictions:  dict,
        regime_info:  Optional[dict] = None,
        multiday:     Optional[dict] = None,
    ) -> bool:
        """
        Send WhatsApp message with predictions.

        Returns True if sent successfully, False otherwise.
        """
        if not self.enabled:
            logger.info("WhatsApp alerts disabled in config")
            return False

        if not self.account_sid or not self.auth_token or not self.to_number:
            logger.warning(
                "WhatsApp not configured. Add to config.yaml:\n"
                "  alerts:\n"
                "    whatsapp:\n"
                "      enabled: true\n"
                "      account_sid: ACxxxx\n"
                "      auth_token: xxxx\n"
                "      from_number: whatsapp:+14155238886\n"
                "      to_number: whatsapp:+91XXXXXXXXXX"
            )
            return False

        try:
            from twilio.rest import Client
        except ImportError:
            logger.error(
                "Twilio not installed. Run: pip install twilio"
            )
            return False

        try:
            body = build_alert_message(
                predictions, regime_info, multiday, format="text"
            )
            # WhatsApp has 1600 char limit — truncate if needed
            if len(body) > 1500:
                body = body[:1500] + "\n\n_...truncated. See email for full report._"

            client  = Client(self.account_sid, self.auth_token)
            message = client.messages.create(
                body=body,
                from_=self.from_number,
                to=self.to_number,
            )
            logger.success(f"WhatsApp sent! SID: {message.sid}")
            return True

        except Exception as e:
            logger.error(f"WhatsApp failed: {e}")
            return False


# ══════════════════════════════════════════════════════════════════════════════
# Combined alert runner
# ══════════════════════════════════════════════════════════════════════════════

def send_daily_alerts(
    predictions:  dict,
    config:       dict,
    regime_info:  Optional[dict] = None,
    multiday:     Optional[dict] = None,
) -> dict:
    """
    Send alerts via all configured channels.

    Parameters
    ----------
    predictions : {ticker: {'direction', 'score', 'price', 'confidence'}}
    config      : loaded config.yaml dict
    regime_info : RegimeDetector output
    multiday    : {ticker: {'3d': 'UP'/'DOWN', '5d': 'UP'/'DOWN'}}

    Returns
    -------
    dict: {'email': bool, 'whatsapp': bool}
    """
    results = {}

    # Email
    email_alerter   = EmailAlerter(config)
    results["email"] = email_alerter.send(predictions, regime_info, multiday)

    # WhatsApp
    wa_alerter          = WhatsAppAlerter(config)
    results["whatsapp"] = wa_alerter.send(predictions, regime_info, multiday)

    return results