## Dependencies

#Install selenium binaries  https://googlechromelabs.github.io/chrome-for-testing/

* chrome	linux64	https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/117.0.5938.149/linux64/chrome-linux64.zip	200
* chrome	mac-arm64	https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/117.0.5938.149/mac-arm64/chrome-mac-arm64.zip	200
* chrome	mac-x64	https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/117.0.5938.149/mac-x64/chrome-mac-x64.zip	200
* chrome	win32	https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/117.0.5938.149/win32/chrome-win32.zip	200
* chrome	win64	https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/117.0.5938.149/win64/chrome-win64.zip	200
* chromedriver	linux64	https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/117.0.5938.149/linux64/chromedriver-linux64.zip	200
* chromedriver	mac-arm64	https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/117.0.5938.149/mac-arm64/chromedriver-mac-arm64.zip	200
* chromedriver	mac-x64	https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/117.0.5938.149/mac-x64/chromedriver-mac-x64.zip	200
* chromedriver	win32	https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/117.0.5938.149/win32/chromedriver-win32.zip	200
* chromedriver	win64	https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/117.0.5938.149/win64/chromedriver-win64.zip	200


## For MacOS

```bash
sudo chmod +x ./chrome-mac-arm64/chromedriver
sudo xattr -r -d com.apple.quarantine /path/to/chrome-mac-arm64/chromedriver
sudo xattr -r -d com.apple.quarantine /path/to/chrome-mac-arm64/GoogleChrome.app
```