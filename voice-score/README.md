# 音声楽譜化

日本語の発話音声を、入力した歌詞と音高の解析から歌詞付きMusicXML楽譜へ変換するブラウザアプリです。

## GitHub Pages

`.github/workflows/deploy-voice-score.yml` が `main` への変更を検知し、Viteでビルドして `/cross-span-tupets/voice-score/` へ公開します。

音素辞書・HuBERT・Whisper・VAD・SwiftF0のモデルは、GitHubリポジトリへ巨大なバイナリを含めず、初回解析時にCDNまたはHugging Faceから取得します。解析にはネットワーク接続が必要です。
