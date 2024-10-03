import os
import argparse
import datetime
import logging
import subprocess  # ffmpegを呼び出すために使用
from faster_whisper import WhisperModel

class MeeTrans:
    def __init__(self, input_file, output_file, model, timestamp, prompt_file=None):
        """MeeTransクラスの初期化メソッド。

        Args:
            input_file (str): 入力する音声または動画ファイルのパス。
            output_file (str): 出力先のテキストファイルのパス。
            model (str): 使用するAIモデル名。例: 'tiny', 'base', 'small', 'medium', 'large-v3'。デフォルトは'large-v3'。
            timestamp (bool): 出力にタイムスタンプを含めるかどうかのフラグ。
            prompt_file (str, optional): 用語集や参考文を記載したテキストファイルのパス。デフォルトはNone。
        """
        self.input_file = input_file
        self.output_file = output_file
        self.model = model
        self.timestamp = timestamp
        self.prompt_file = prompt_file
        self.audio_file = None  # 動画ファイルを音声ファイルに変換する場合に一時的に使う
        self.setup_logging()

        # Whisperモデルのインスタンスを作成
        cwd = os.path.dirname(__file__).replace('\\', '/')
        model_path = f'{cwd}/models'
        os.makedirs(model_path, exist_ok=True)
        self.whisper_model = WhisperModel(model, device="cuda", compute_type="float16", download_root=model_path)

    def setup_logging(self):
        """ロギングの設定を行うメソッド。

        ログレベルをINFOに設定し、ログフォーマットを定義します。ロガーを
        インスタンス変数`self.logger`に格納します。
        """
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def convert_video_to_audio(self):
        """動画ファイルをmp3に変換するメソッド。

        入力されたファイルが動画の場合、ffmpegを使用して音声ファイル（mp3形式）に変換します。

        Returns:
            str: 変換された音声ファイル（mp3）のパス。
        """
        audio_file = f"{os.path.splitext(self.input_file)[0]}.mp3"
        self.logger.info(f"[MeeTrans.convert_video_to_audio] 動画ファイル '{self.input_file}' をmp3に変換中...")

        command = ["ffmpeg", "-i", self.input_file, "-q:a", "0", "-map", "a", audio_file, "-y", "-loglevel", "quiet"]
        
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self.logger.info(f"[MeeTrans.convert_video_to_audio] 音声ファイル '{audio_file}' に変換完了。")
        return audio_file

    def transcribe(self):
        """音声/動画ファイルを文字起こしするメソッド。

        入力ファイルが動画の場合は、先に`convert_video_to_audio()`で音声ファイルに変換し、
        その後Whisperモデルを使用して文字起こしを行います。結果は指定されたテキストファイルに
        保存されます。
        """
        # 入力ファイルの拡張子を確認し、動画ファイルの場合は音声に変換
        ext = os.path.splitext(self.input_file)[1].lower()
        if ext in ['.mp4', '.mkv', '.mov', '.avi']:
            self.audio_file = self.convert_video_to_audio()
        else:
            self.audio_file = self.input_file

        self.logger.info(f"[MeeTrans.transcribe] 音声ファイル '{self.audio_file}' をモデル '{self.model}' で文字起こし中...")

        # プロンプトファイルが指定されている場合、内容を読み込む
        initial_prompt = None
        if self.prompt_file:
            self.logger.info(f"[MeeTrans.transcribe] プロンプトファイル '{self.prompt_file}' を読み込んでいます...")
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                initial_prompt = f.read()

        # 文字起こしを実行
        segments, _ = self.whisper_model.transcribe(
            self.audio_file,
            language="ja",
            vad_filter=True,
            initial_prompt=initial_prompt
        )

        self.save_transcription(segments)

    def format_timestamp(self, seconds):
        """秒数を [hh:mm:ss] 形式にフォーマットするメソッド。

        Args:
            seconds (float): タイムスタンプの秒数。

        Returns:
            str: [hh:mm:ss] 形式にフォーマットされたタイムスタンプ。
        """
        seconds = int(seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        sec = int(seconds % 60)
        return f'{hours:02}:{minutes:02}:{sec:02}'

    def save_transcription(self, segments):
        """文字起こし結果をファイルに保存するメソッド。

        Args:
            segments (list): Whisperモデルからの文字起こしセグメントのリスト。
        """
        with open(self.output_file, "w", encoding="utf-8") as f:
            for segment in segments:
                if self.timestamp:
                    start = self.format_timestamp(segment.start)
                    end = self.format_timestamp(segment.end)
                    f.write(f'[{start} -> {end}] {segment.text}\n')
                else:
                    f.write(f'{segment.text}\n')
        self.logger.info(f"[MeeTrans.save_transcription] 文字起こし結果を '{self.output_file}' に保存しました。")

    @staticmethod
    def parse_arguments():
        """コマンドライン引数を解析する静的メソッド。

        使用可能なモデルのリストを提供し、コマンドライン引数を解析します。

        Returns:
            argparse.Namespace: 解析された引数。
        """
        today_str = datetime.datetime.now().strftime('%Y%m%d')
        default_output = f"meetrans_output_{today_str}.txt"
        available_models = ['tiny', 'base', 'small', 'medium', 'large-v3']

        parser = argparse.ArgumentParser(prog='meetrans', description='音声/動画ファイルをAIで文字起こししてテキストファイルに保存するプログラム')
        parser.add_argument('input_file', type=str, help='入力する音声または動画ファイルのパス')
        parser.add_argument('--output', '-o', type=str, default=default_output, 
                            help=f'出力先のテキストファイルのパス（デフォルトは "{default_output}"）')
        parser.add_argument('--model', '-m', type=str, choices=available_models, default='large-v3',
                            help=f"使用するAIモデルを指定（デフォルトは 'large-v3'）")
        parser.add_argument('--prompt', '-p', type=str, help='用語集や参考文を記載したテキストファイルのパス（任意）')
        parser.add_argument('--timestamp', '-t', action='store_true', help='出力にタイムスタンプを含めるかどうか')

        return parser.parse_args()

def main():
    """メイン処理を実行する関数。

    コマンドライン引数を解析し、MeeTransクラスのインスタンスを作成し、
    文字起こしを実行します。
    """
    args = MeeTrans.parse_arguments()
    transcriber = MeeTrans(args.input_file, args.output, args.model, args.timestamp, args.prompt)
    transcriber.transcribe()

if __name__ == '__main__':
    main()
