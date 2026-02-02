import yt_dlp
import shutil
import imageio_ffmpeg
import subprocess

def test_ytdlp():
    url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    print(f"Testing yt-dlp with URL: {url}")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Extracting info...")
            info = ydl.extract_info(url, download=False)
            stream_url = info['url']
            print(f"SUCCESS: Got stream URL: {stream_url[:50]}...")
            
            # Now test ffmpeg access
            ffmpeg_exe = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
            print(f"FFMPEG Path: {ffmpeg_exe}")
            
            # Test ffmpeg execution on this URL (just probe it)
            print("Probing stream with ffprobe/ffmpeg...")
            
            cmd = [
                ffmpeg_exe,
                '-y',
                '-i', stream_url,
                '-t', '1', # 1 second
                '-f', 'null',
                '-'
            ]
            
            process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print("FFMPEG Probe SUCCESS")
            else:
                print(f"FFMPEG Probe FAILED. Return code: {process.returncode}")
                print(f"Stderr: {stderr.decode()[:500]}") # Print first 500 chars
            
    except Exception as e:
        print(f"yt-dlp FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ytdlp()
