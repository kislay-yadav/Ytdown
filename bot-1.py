import os
import sys
import re
import asyncio
import logging
import shutil
import uuid
import aiohttp
import yt_dlp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

BOT_TOKEN: Optional[str] = os.getenv("BOT_TOKEN")
OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"

TELEGRAM_FILE_LIMIT = 50 * 1024 * 1024
DOWNLOAD_DIR = "/tmp/downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

VIDEO_QUALITIES = {
    "144p": "bestvideo[height<=144]+bestaudio/best[height<=144]",
    "240p": "bestvideo[height<=240]+bestaudio/best[height<=240]",
    "360p": "bestvideo[height<=360]+bestaudio/best[height<=360]",
    "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]",
    "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
    "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
    "1440p": "bestvideo[height<=1440]+bestaudio/best[height<=1440]",
    "2160p": "bestvideo[height<=2160]+bestaudio/best[height<=2160]",
}

AUDIO_QUALITIES = {
    "64kbps": "64",
    "128kbps": "128",
    "192kbps": "192",
    "256kbps": "256",
    "320kbps": "320",
}

RATE_LIMIT_SECONDS = 5
MAX_RETRIES = 3
MAX_CONCURRENT_DOWNLOADS = 5

YOUTUBE_URL_PATTERNS = [
    r'(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+',
    r'(https?://)?(www\.)?youtube\.com/shorts/[\w-]+',
    r'(https?://)?(www\.)?youtu\.be/[\w-]+',
    r'(https?://)?(www\.)?youtube\.com/v/[\w-]+',
    r'(https?://)?m\.youtube\.com/watch\?v=[\w-]+',
]

download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
user_rate_limits: Dict[int, datetime] = {}


class AIHelper:
    def __init__(self):
        self.enabled = bool(OPENROUTER_API_KEY)
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://telegram-yt-bot.replit.app",
            "X-Title": "YouTube Downloader Bot",
        }
    
    async def get_response(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        if not self.enabled:
            return None
        
        if system_prompt is None:
            system_prompt = """You are a friendly, helpful assistant for a YouTube downloader Telegram bot. 
Keep responses concise, use emojis appropriately, and be helpful. 
Never reveal technical details or errors directly - explain them in simple terms."""
        
        try:
            payload = {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    OPENROUTER_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        logger.warning(f"AI API error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"AI request failed: {e}")
            return None
    
    async def explain_error(self, error_type: str) -> str:
        prompts = {
            "private": "Explain briefly that a YouTube video is private and can't be downloaded. Keep it friendly with emoji.",
            "age_restricted": "Explain that a video is age-restricted and may require login. Suggest alternatives. Use emoji.",
            "unavailable": "Explain that a video is unavailable or deleted. Be sympathetic. Use emoji.",
            "too_large": "Explain that a file is too large for Telegram (>50MB) and suggest lower quality. Use emoji.",
            "network": "Explain there was a temporary network issue and they should try again. Use emoji.",
            "invalid_url": "Explain that the URL doesn't look like a valid YouTube link. Give examples of valid formats. Use emoji.",
            "rate_limit": "Explain kindly that they're sending requests too fast and should wait a moment. Use emoji.",
            "busy": "Explain kindly that the bot is busy processing other downloads and they should try again in a moment. Use emoji.",
        }
        
        prompt = prompts.get(error_type, f"Explain this error in simple terms: {error_type}")
        response = await self.get_response(prompt)
        
        if response:
            return response
        
        fallbacks = {
            "private": "🔒 This video is private and can't be downloaded. The owner has restricted access.",
            "age_restricted": "🔞 This video is age-restricted. Unfortunately, these videos require login verification.",
            "unavailable": "😔 This video is no longer available. It may have been removed by the uploader.",
            "too_large": "📦 The file is too large for Telegram (>50MB). Try selecting a lower quality option.",
            "network": "🌐 There was a network hiccup. Please try again in a moment!",
            "invalid_url": "🔗 That doesn't look like a valid YouTube URL. Please send a link like:\nyoutube.com/watch?v=...\nor youtu.be/...",
            "rate_limit": "⏳ Whoa, slow down! Please wait a few seconds before your next request.",
            "busy": "⏳ I'm currently processing other downloads. Please try again in a moment!",
        }
        
        return fallbacks.get(error_type, "❌ Something went wrong. Please try again!")
    
    async def chat(self, message: str) -> str:
        response = await self.get_response(
            message,
            system_prompt="""You are a helpful assistant for a YouTube downloader bot. 
Answer questions about the bot's features, help with issues, and be friendly.
Features: Download videos (144p-4K), download audio (64-320kbps), handle Shorts.
Keep responses concise and use emojis appropriately."""
        )
        return response if response else "I'm here to help! You can send me a YouTube link to download videos or audio. Use the menu buttons to get started!"


class BotUI:
    
    @staticmethod
    def main_menu() -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("🎥 Download Video", callback_data="mode_video"),
                InlineKeyboardButton("🎧 Download Audio", callback_data="mode_audio"),
            ],
            [
                InlineKeyboardButton("🤖 AI Help", callback_data="ai_help"),
                InlineKeyboardButton("ℹ️ About", callback_data="about"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def back_to_menu() -> InlineKeyboardMarkup:
        keyboard = [[InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]]
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def video_quality_menu(available_formats: List[str]) -> InlineKeyboardMarkup:
        quality_emojis = {
            "144p": "📱", "240p": "📱", "360p": "📺", "480p": "📺",
            "720p": "🖥️", "1080p": "🖥️", "1440p": "🎬", "2160p": "🎬"
        }
        
        keyboard = []
        row = []
        for quality in available_formats:
            emoji = quality_emojis.get(quality, "🎥")
            label = quality.replace("2160p", "4K").replace("1440p", "2K")
            row.append(InlineKeyboardButton(f"{emoji} {label}", callback_data=f"vq_{quality}"))
            if len(row) == 2:
                keyboard.append(row)
                row = []
        if row:
            keyboard.append(row)
        
        keyboard.append([InlineKeyboardButton("🔙 Cancel", callback_data="cancel")])
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def audio_quality_menu() -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("🔈 64kbps", callback_data="aq_64kbps"),
                InlineKeyboardButton("🔉 128kbps", callback_data="aq_128kbps"),
            ],
            [
                InlineKeyboardButton("🔊 192kbps", callback_data="aq_192kbps"),
                InlineKeyboardButton("🔊 256kbps", callback_data="aq_256kbps"),
            ],
            [
                InlineKeyboardButton("🎵 320kbps (Best)", callback_data="aq_320kbps"),
            ],
            [InlineKeyboardButton("🔙 Cancel", callback_data="cancel")],
        ]
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def progress_bar(progress: float, width: int = 20) -> str:
        filled = int(width * progress / 100)
        empty = width - filled
        bar = "█" * filled + "░" * empty
        return f"[{bar}] {progress:.1f}%"
    
    @staticmethod
    def format_size(size_bytes: int) -> str:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    
    @staticmethod
    def format_duration(seconds: int) -> str:
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"
    
    @staticmethod
    def welcome_message() -> str:
        return """
🎬 *YouTube Downloader Bot*

Welcome! I can download YouTube videos and audio for you.

*Features:*
• 🎥 Video downloads (144p to 4K)
• 🎧 Audio extraction (64-320kbps)
• ⚡ Fast & reliable downloads
• 🎯 YouTube Shorts support
• 🤖 AI-powered assistance

*How to use:*
1. Choose Video or Audio mode
2. Send me a YouTube link
3. Select quality
4. Get your file!

Use the buttons below to get started!
"""
    
    @staticmethod
    def about_message() -> str:
        return """
ℹ️ *About This Bot*

*YouTube Downloader Bot v1.0*

A premium Telegram bot for downloading YouTube content.

*Capabilities:*
• Download videos up to 4K quality
• Extract audio in MP3 format
• Handle YouTube Shorts
• Smart quality selection
• AI-powered assistance

*Limits:*
• Max file size: 50MB (Telegram limit)
• Supports public videos only

*Powered by:*
• yt-dlp for downloads
• FFmpeg for processing
• OpenRouter AI for assistance

Made with ❤️ for the community
"""
    
    @staticmethod
    def processing_message(state: str, details: str = "") -> str:
        states = {
            "analyzing": "🔍 *Analyzing link...*\nPlease wait while I fetch video information.",
            "fetching": "📊 *Fetching qualities...*\nGetting available format options.",
            "downloading": f"⬇️ *Downloading...*\n{details}",
            "converting": "🎛️ *Converting...*\nProcessing your file.",
            "uploading": "✅ *Uploading...*\nSending file to Telegram.",
            "complete": "🎉 *Done!*\nYour file is ready.",
            "error": f"❌ *Error*\n{details}",
        }
        return states.get(state, f"⏳ Processing... {details}")
    
    @staticmethod
    def video_info_message(info: Dict) -> str:
        title = info.get("title", "Unknown")
        channel = info.get("channel", "Unknown")
        duration = info.get("duration", 0)
        views = info.get("view_count", 0)
        is_short = info.get("is_short", False)
        
        duration_str = BotUI.format_duration(duration) if duration else "Unknown"
        views_str = f"{views:,}" if views else "Unknown"
        video_type = "📱 YouTube Short" if is_short else "🎬 YouTube Video"
        
        return f"""
{video_type}

*{title}*

👤 Channel: {channel}
⏱️ Duration: {duration_str}
👁️ Views: {views_str}

Select your preferred quality below:
"""
    
    @staticmethod
    def download_complete_message(filename: str, size: int, quality: str, is_video: bool) -> str:
        media_emoji = "🎥" if is_video else "🎧"
        media_type = "Video" if is_video else "Audio"
        size_str = BotUI.format_size(size)
        
        return f"""
{media_emoji} *{media_type} Downloaded!*

📁 Quality: {quality}
💾 Size: {size_str}

Enjoy your download! 🎉
"""


class YouTubeDownloader:
    def __init__(self):
        self.download_dir = DOWNLOAD_DIR
        os.makedirs(self.download_dir, exist_ok=True)
    
    def _get_user_download_dir(self, user_id: int, request_id: str) -> str:
        user_dir = os.path.join(self.download_dir, f"user_{user_id}", request_id)
        os.makedirs(user_dir, exist_ok=True)
        return user_dir
    
    def is_valid_youtube_url(self, url: str) -> bool:
        for pattern in YOUTUBE_URL_PATTERNS:
            if re.match(pattern, url):
                return True
        return False
    
    def sanitize_filename(self, filename: str) -> str:
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        sanitized = sanitized[:100] if len(sanitized) > 100 else sanitized
        return sanitized or "download"
    
    async def get_video_info(self, url: str) -> Tuple[Optional[Dict], Optional[str]]:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': True,
        }
        
        try:
            loop = asyncio.get_event_loop()
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = await loop.run_in_executor(None, lambda: ydl.extract_info(url, download=False))
            
            if not info:
                return None, "unavailable"
            
            is_short = '/shorts/' in url or info.get('duration', 0) <= 60
            
            formats = info.get('formats', [])
            available_qualities = self._get_available_qualities(formats)
            
            video_info = {
                'id': info.get('id'),
                'title': info.get('title', 'Unknown'),
                'channel': info.get('channel', info.get('uploader', 'Unknown')),
                'duration': info.get('duration', 0),
                'view_count': info.get('view_count', 0),
                'thumbnail': info.get('thumbnail'),
                'is_short': is_short,
                'available_video_qualities': available_qualities,
                'url': url,
                'formats': formats,
            }
            
            return video_info, None
            
        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e).lower()
            if 'private' in error_msg:
                return None, "private"
            elif 'age' in error_msg or 'sign in' in error_msg:
                return None, "age_restricted"
            elif 'unavailable' in error_msg or 'removed' in error_msg:
                return None, "unavailable"
            else:
                logger.error(f"Download error: {e}")
                return None, "network"
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return None, "network"
    
    def _get_available_qualities(self, formats: List[Dict]) -> List[str]:
        available = set()
        quality_heights = {
            144: "144p", 240: "240p", 360: "360p", 480: "480p",
            720: "720p", 1080: "1080p", 1440: "1440p", 2160: "2160p"
        }
        
        for fmt in formats:
            height = fmt.get('height')
            if height:
                for h, q in quality_heights.items():
                    if height >= h:
                        available.add(q)
        
        quality_order = ["144p", "240p", "360p", "480p", "720p", "1080p", "1440p", "2160p"]
        return [q for q in quality_order if q in available]
    
    async def download_video(
        self,
        url: str,
        quality: str,
        user_id: int,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        
        request_id = str(uuid.uuid4())[:8]
        user_download_dir = self._get_user_download_dir(user_id, request_id)
        
        async with download_semaphore:
            logger.info(f"Starting video download for user {user_id}, request {request_id}")
            
            format_spec = VIDEO_QUALITIES.get(quality, "bestvideo+bestaudio/best")
            
            info, error = await self.get_video_info(url)
            if error:
                self.cleanup_user_dir(user_download_dir)
                return None, error
            
            filename = self.sanitize_filename(info['title'])
            
            ydl_opts = {
                'format': format_spec,
                'outtmpl': os.path.join(user_download_dir, f"{filename}.%(ext)s"),
                'merge_output_format': 'mp4',
                'quiet': True,
                'no_warnings': True,
                'retries': MAX_RETRIES,
                'fragment_retries': MAX_RETRIES,
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',
                }],
            }
            
            try:
                loop = asyncio.get_running_loop()
                
                if progress_callback:
                    ydl_opts['progress_hooks'] = [self._create_progress_hook(progress_callback, loop)]
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    await loop.run_in_executor(None, lambda: ydl.download([url]))
                
                final_path = self._find_downloaded_file(user_download_dir, filename, ['.mp4', '.mkv', '.webm'])
                
                if final_path and os.path.exists(final_path):
                    file_size = os.path.getsize(final_path)
                    logger.info(f"Downloaded video for user {user_id}: {final_path}, size: {file_size} bytes")
                    if file_size > TELEGRAM_FILE_LIMIT:
                        logger.warning(f"File too large for user {user_id}: {file_size} > {TELEGRAM_FILE_LIMIT}")
                        self.cleanup_user_dir(user_download_dir)
                        return None, "too_large"
                    return final_path, None
                
                logger.error(f"Download completed but file not found for user {user_id}: {filename}")
                self.cleanup_user_dir(user_download_dir)
                return None, "network"
                
            except yt_dlp.utils.DownloadError as e:
                error_msg = str(e).lower()
                logger.error(f"yt-dlp download error for user {user_id}: {e}")
                self.cleanup_user_dir(user_download_dir)
                if 'private' in error_msg:
                    return None, "private"
                elif 'age' in error_msg or 'sign in' in error_msg:
                    return None, "age_restricted"
                elif 'unavailable' in error_msg or 'removed' in error_msg:
                    return None, "unavailable"
                return None, "network"
            except Exception as e:
                logger.error(f"Download failed for user {user_id} with exception: {type(e).__name__}: {e}")
                self.cleanup_user_dir(user_download_dir)
                return None, "network"
    
    async def download_audio(
        self,
        url: str,
        quality: str,
        user_id: int,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        
        request_id = str(uuid.uuid4())[:8]
        user_download_dir = self._get_user_download_dir(user_id, request_id)
        
        async with download_semaphore:
            logger.info(f"Starting audio download for user {user_id}, request {request_id}")
            
            bitrate = AUDIO_QUALITIES.get(quality, "192")
            
            info, error = await self.get_video_info(url)
            if error:
                self.cleanup_user_dir(user_download_dir)
                return None, error
            
            filename = self.sanitize_filename(info['title'])
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(user_download_dir, f"{filename}.%(ext)s"),
                'quiet': True,
                'no_warnings': True,
                'retries': MAX_RETRIES,
                'fragment_retries': MAX_RETRIES,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': bitrate,
                }],
            }
            
            try:
                loop = asyncio.get_running_loop()
                
                if progress_callback:
                    ydl_opts['progress_hooks'] = [self._create_progress_hook(progress_callback, loop)]
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    await loop.run_in_executor(None, lambda: ydl.download([url]))
                
                final_path = os.path.join(user_download_dir, f"{filename}.mp3")
                
                if os.path.exists(final_path):
                    file_size = os.path.getsize(final_path)
                    logger.info(f"Downloaded audio for user {user_id}: {final_path}, size: {file_size} bytes")
                    if file_size > TELEGRAM_FILE_LIMIT:
                        logger.warning(f"File too large for user {user_id}: {file_size} > {TELEGRAM_FILE_LIMIT}")
                        self.cleanup_user_dir(user_download_dir)
                        return None, "too_large"
                    return final_path, None
                
                logger.error(f"Audio download completed but file not found for user {user_id}: {filename}.mp3")
                self.cleanup_user_dir(user_download_dir)
                return None, "network"
                
            except yt_dlp.utils.DownloadError as e:
                error_msg = str(e).lower()
                logger.error(f"yt-dlp audio download error for user {user_id}: {e}")
                self.cleanup_user_dir(user_download_dir)
                if 'private' in error_msg:
                    return None, "private"
                elif 'age' in error_msg or 'sign in' in error_msg:
                    return None, "age_restricted"
                elif 'unavailable' in error_msg or 'removed' in error_msg:
                    return None, "unavailable"
                return None, "network"
            except Exception as e:
                logger.error(f"Audio download failed for user {user_id} with exception: {type(e).__name__}: {e}")
                self.cleanup_user_dir(user_download_dir)
                return None, "network"
    
    def _find_downloaded_file(self, directory: str, base_name: str, extensions: List[str]) -> Optional[str]:
        for ext in extensions:
            path = os.path.join(directory, f"{base_name}{ext}")
            if os.path.exists(path):
                return path
        
        for file in os.listdir(directory):
            if file.startswith(base_name):
                return os.path.join(directory, file)
        
        return None
    
    def _create_progress_hook(self, callback: Callable, loop: asyncio.AbstractEventLoop):
        def hook(d):
            if d['status'] == 'downloading':
                total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                downloaded = d.get('downloaded_bytes', 0)
                if total > 0:
                    progress = (downloaded / total) * 100
                    loop.call_soon_threadsafe(lambda p=progress: asyncio.create_task(callback(p)))
            elif d['status'] == 'finished':
                loop.call_soon_threadsafe(lambda: asyncio.create_task(callback(100)))
        return hook
    
    def cleanup_file(self, filepath: str):
        try:
            if filepath and os.path.exists(filepath):
                parent_dir = os.path.dirname(filepath)
                os.remove(filepath)
                if parent_dir != self.download_dir and os.path.exists(parent_dir):
                    try:
                        shutil.rmtree(parent_dir)
                    except:
                        pass
        except Exception as e:
            logger.warning(f"Failed to cleanup file {filepath}: {e}")
    
    def cleanup_user_dir(self, user_dir: str):
        try:
            if user_dir and os.path.exists(user_dir) and user_dir != self.download_dir:
                shutil.rmtree(user_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup user directory {user_dir}: {e}")


ai_helper = AIHelper()
ui = BotUI()
downloader = YouTubeDownloader()


def get_user_state(context: ContextTypes.DEFAULT_TYPE) -> Dict:
    if 'state' not in context.user_data:
        context.user_data['state'] = {
            'mode': None,
            'url': None,
            'video_info': None,
            'quality': None,
            'awaiting_url': False,
            'awaiting_ai_chat': False,
        }
    return context.user_data['state']


def check_rate_limit(user_id: int) -> bool:
    now = datetime.now()
    if user_id in user_rate_limits:
        if now - user_rate_limits[user_id] < timedelta(seconds=RATE_LIMIT_SECONDS):
            return False
    user_rate_limits[user_id] = now
    return True


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['state'] = {
        'mode': None,
        'url': None,
        'video_info': None,
        'quality': None,
        'awaiting_url': False,
        'awaiting_ai_chat': False,
    }
    
    await update.message.reply_text(
        ui.welcome_message(),
        parse_mode='Markdown',
        reply_markup=ui.main_menu()
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        ui.welcome_message(),
        parse_mode='Markdown',
        reply_markup=ui.main_menu()
    )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    state = get_user_state(context)
    data = query.data
    
    if data == "main_menu":
        state['mode'] = None
        state['url'] = None
        state['awaiting_url'] = False
        state['awaiting_ai_chat'] = False
        await query.edit_message_text(
            ui.welcome_message(),
            parse_mode='Markdown',
            reply_markup=ui.main_menu()
        )
    
    elif data == "mode_video":
        state['mode'] = 'video'
        state['awaiting_url'] = True
        state['awaiting_ai_chat'] = False
        await query.edit_message_text(
            "🎥 *Video Download Mode*\n\nSend me a YouTube link to download as video.",
            parse_mode='Markdown',
            reply_markup=ui.back_to_menu()
        )
    
    elif data == "mode_audio":
        state['mode'] = 'audio'
        state['awaiting_url'] = True
        state['awaiting_ai_chat'] = False
        await query.edit_message_text(
            "🎧 *Audio Download Mode*\n\nSend me a YouTube link to download as MP3.",
            parse_mode='Markdown',
            reply_markup=ui.back_to_menu()
        )
    
    elif data == "ai_help":
        state['awaiting_ai_chat'] = True
        state['awaiting_url'] = False
        await query.edit_message_text(
            "🤖 *AI Assistant*\n\nAsk me anything about the bot or how to use it!",
            parse_mode='Markdown',
            reply_markup=ui.back_to_menu()
        )
    
    elif data == "about":
        await query.edit_message_text(
            ui.about_message(),
            parse_mode='Markdown',
            reply_markup=ui.back_to_menu()
        )
    
    elif data.startswith("vq_"):
        quality = data[3:]
        state['quality'] = quality
        await process_download(query, context, user_id, state, is_video=True)
    
    elif data.startswith("aq_"):
        quality = data[3:]
        state['quality'] = quality
        await process_download(query, context, user_id, state, is_video=False)
    
    elif data == "cancel":
        state['mode'] = None
        state['url'] = None
        state['quality'] = None
        state['awaiting_url'] = False
        await query.edit_message_text(
            "❌ *Cancelled*\n\nOperation cancelled. Use the menu to start again.",
            parse_mode='Markdown',
            reply_markup=ui.main_menu()
        )


async def process_download(query, context: ContextTypes.DEFAULT_TYPE, user_id: int, state: Dict, is_video: bool):
    url = state.get('url')
    quality = state.get('quality')
    
    if not url or not quality:
        await query.edit_message_text(
            "❌ Missing information. Please start again.",
            reply_markup=ui.main_menu()
        )
        return
    
    status_msg = await query.edit_message_text(
        ui.processing_message("downloading", ui.progress_bar(0)),
        parse_mode='Markdown'
    )
    
    last_update: list[float] = [0.0]
    
    async def progress_callback(progress: float):
        if progress - last_update[0] >= 10:
            last_update[0] = float(progress)
            try:
                await status_msg.edit_text(
                    ui.processing_message("downloading", ui.progress_bar(progress)),
                    parse_mode='Markdown'
                )
            except:
                pass
    
    try:
        if is_video:
            filepath, error = await downloader.download_video(url, quality, user_id, progress_callback)
        else:
            filepath, error = await downloader.download_audio(url, quality, user_id, progress_callback)
        
        if error:
            error_msg = await ai_helper.explain_error(error)
            await status_msg.edit_text(
                f"❌ *Download Failed*\n\n{error_msg}",
                parse_mode='Markdown',
                reply_markup=ui.main_menu()
            )
            return
        
        await status_msg.edit_text(
            ui.processing_message("uploading"),
            parse_mode='Markdown'
        )
        
        if not filepath:
            raise ValueError("Download failed - no file path returned")
        
        file_size = os.path.getsize(filepath)
        filename = os.path.basename(filepath)
        
        with open(filepath, 'rb') as f:
            if is_video:
                await query.message.reply_video(
                    video=f,
                    caption=ui.download_complete_message(filename, file_size, quality, True),
                    parse_mode='Markdown',
                    supports_streaming=True
                )
            else:
                await query.message.reply_audio(
                    audio=f,
                    caption=ui.download_complete_message(filename, file_size, quality, False),
                    parse_mode='Markdown'
                )
        
        await status_msg.edit_text(
            "✅ *Download Complete!*\n\nYour file has been sent above.",
            parse_mode='Markdown',
            reply_markup=ui.main_menu()
        )
        
        downloader.cleanup_file(filepath)
        
    except Exception as e:
        logger.error(f"Download error for user {user_id}: {e}")
        error_msg = await ai_helper.explain_error("network")
        await status_msg.edit_text(
            f"❌ *Error*\n\n{error_msg}",
            parse_mode='Markdown',
            reply_markup=ui.main_menu()
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text
    state = get_user_state(context)
    
    if not check_rate_limit(user_id):
        error_msg = await ai_helper.explain_error("rate_limit")
        await update.message.reply_text(error_msg)
        return
    
    if state.get('awaiting_ai_chat'):
        response = await ai_helper.chat(text)
        await update.message.reply_text(
            f"🤖 {response}",
            reply_markup=ui.back_to_menu()
        )
        return
    
    if downloader.is_valid_youtube_url(text):
        if not state.get('mode'):
            await update.message.reply_text(
                "🎬 *I found a YouTube link!*\n\nWhat would you like to download?",
                parse_mode='Markdown',
                reply_markup=ui.main_menu()
            )
            state['url'] = text
            return
        
        state['url'] = text
        
        status_msg = await update.message.reply_text(
            ui.processing_message("analyzing"),
            parse_mode='Markdown'
        )
        
        video_info, error = await downloader.get_video_info(text)
        
        if error:
            error_msg = await ai_helper.explain_error(error)
            await status_msg.edit_text(
                f"❌ *Error*\n\n{error_msg}",
                parse_mode='Markdown',
                reply_markup=ui.main_menu()
            )
            return
        
        state['video_info'] = video_info
        
        if state['mode'] == 'video' and video_info:
            available = video_info.get('available_video_qualities', ['720p'])
            await status_msg.edit_text(
                ui.video_info_message(video_info),
                parse_mode='Markdown',
                reply_markup=ui.video_quality_menu(available)
            )
        elif video_info:
            await status_msg.edit_text(
                ui.video_info_message(video_info),
                parse_mode='Markdown',
                reply_markup=ui.audio_quality_menu()
            )
    
    elif state.get('awaiting_url'):
        error_msg = await ai_helper.explain_error("invalid_url")
        await update.message.reply_text(
            f"❌ {error_msg}",
            reply_markup=ui.back_to_menu()
        )
    
    else:
        await update.message.reply_text(
            "👋 Hi! Use the menu to get started.",
            reply_markup=ui.main_menu()
        )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error {context.error}")
    
    if update and update.effective_message:
        error_msg = await ai_helper.explain_error("network")
        try:
            await update.effective_message.reply_text(
                f"❌ {error_msg}",
                reply_markup=ui.main_menu()
            )
        except:
            pass


def validate_config() -> bool:
    if not BOT_TOKEN:
        print("ERROR: BOT_TOKEN environment variable is not set!")
        return False
    if not OPENROUTER_API_KEY:
        print("WARNING: OPENROUTER_API_KEY not set. AI features will be disabled.")
    return True


def main():
    if not validate_config():
        logger.error("Configuration validation failed!")
        sys.exit(1)
    
    logger.info("Starting YouTube Downloader Bot...")
    logger.info(f"Max concurrent downloads: {MAX_CONCURRENT_DOWNLOADS}")
    
    application = Application.builder().token(BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    application.add_error_handler(error_handler)
    
    logger.info("Bot is running! Press Ctrl+C to stop.")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
