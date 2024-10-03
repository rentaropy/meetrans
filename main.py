import os
import argparse
import datetime
import logging
from faster_whisper import WhisperModel

class MeeTrans:
    def __init__(self, audio_file, output_file, model, timestamp, prompt_file=None):
        """MeeTransクラスの初期化メソッド。

        Args:
            audio_file (str): 入力するオーディオファイルのパス。
            output_file (str): 出力先のテキストファイルのパス。
            model (str, optional): 使用するAIモデル名。デフォルトは'large-v3'。
            timestamp (bool): タイムスタンプを出力に含めるかどうかのフラグ。
            prompt_file (str, optional): 用語集や参考文を記載したテキストファイルのパス。デフォルトは指定なし。
        """
        self.audio_file = audio_file
        self.output_file = output_file
        self.model = model
        self.timestamp = timestamp  # タイムスタンプのフラグ
        self.prompt_file = prompt_file
        self.setup_logging()
        
        # Whisperモデルのインスタンスを作成
        model_path = './models'
        os.makedirs(model_path, exist_ok=True)
        self.whisper_model = WhisperModel(model, device="cuda", compute_type="float16", download_root=model_path)

    def setup_logging(self):
        """ロギングの設定を行うメソッド。

        ログのレベルとフォーマットを設定し、ロガーのインスタンスを生成します。
        """
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def transcribe(self):
        """音声ファイルを文字起こしするメソッド。

        音声ファイルを指定されたモデルで文字起こしし、結果を
        テキストファイルとして保存します。
        """
        # コマンドライン引数の確認
        self.logger.info(f"[MeeTrans.transcribe] 引数確認: audio_file='{self.audio_file}', output_file='{self.output_file}', model='{self.model}', prompt_file='{self.prompt_file}', timestamp={self.timestamp}")

        self.logger.info(f"[MeeTrans.transcribe] 音声ファイル '{self.audio_file}' をモデル '{self.model}' で文字起こし中...")

        # プロンプトファイルが指定されている場合、内容を読み込む
        initial_prompt = None
        if self.prompt_file:
            self.logger.info(f"[MeeTrans.transcribe] プロンプトファイル '{self.prompt_file}' を読み込んでいます...")
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                initial_prompt = f.read()  # プロンプトの内容を単一の文字列として取得

        # 文字起こしを行い、結果を取得
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
            seconds (float): 秒数。

        Returns:
            str: フォーマットされた時間を表す文字列。
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
                # タイムスタンプを含めるかどうかに応じて出力内容を決定
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
        # 現在の日付をyyyymmdd形式で取得
        today_str = datetime.datetime.now().strftime('%Y%m%d')

        # デフォルトの出力ファイルパス
        default_output = f"meetrans_output_{today_str}.txt"

        # 使用可能なモデルのリスト
        available_models = ['tiny', 'base', 'small', 'medium', 'large-v3']

        # ArgumentParserオブジェクトの作成
        parser = argparse.ArgumentParser(prog='meetrans', description='AIモデルで音声ファイルを文字起こしし、テキストファイルとして保存するプログラム')

        # 第1引数: 音声ファイルのパス (必須)
        parser.add_argument('audio_file', type=str, help='入力するオーディオファイルのパス')

        # オプション: 出力先のテキストファイルのパス (任意、デフォルトはカレントディレクトリの "meetrans_output_yyyymmdd.txt")
        parser.add_argument('--output', '-o', type=str, default=default_output, 
                            help=f'出力先のテキストファイルのパス（指定がない場合はカレントディレクトリの "{default_output}" に保存されます）')

        # オプション: 使用するAIモデルの指定 (任意、デフォルトは'large-v3')
        parser.add_argument('--model', '-m', type=str, choices=available_models, default='large-v3',
                            help=f"使用するAIモデルを指定（オプション）。例: {', '.join(available_models)} (指定がない場合は 'large-v3' が使用されます)")

        # オプション: 用語集や参考文を記載したテキストファイルのパス (任意、デフォルトは指定なし)
        parser.add_argument('--prompt', '-p', type=str, 
                            help='用語集や参考文を記載したテキストファイルのパス（オプション）。デフォルトは指定なし。')

        # オプション: タイムスタンプを含めるかどうか (任意、デフォルトは含める)
        parser.add_argument('--timestamp', '-t', action='store_true', 
                            help='出力にタイムスタンプを含めるかどうか（指定がない場合は含まれます）')

        # 引数をパース
        return parser.parse_args()


def main():
    """メイン処理を実行する関数。

    コマンドライン引数を解析し、MeeTransクラスのインスタンスを作成し、
    文字起こしを実行します。
    """
    # 引数を解析
    args = MeeTrans.parse_arguments()

    # MeeTransクラスのインスタンスを作成し、文字起こしを実行
    transcriber = MeeTrans(args.audio_file, args.output, args.model, args.timestamp, args.prompt)
    transcriber.transcribe()


if __name__ == '__main__':
    main()
