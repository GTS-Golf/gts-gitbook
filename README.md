# 공지사항

안녕하세요. GTS골프 입니다.

GTS에서 제공하는 A/S메뉴얼 페이지 입니다.



깃북의 주소는 https://tensorflowkorea.gitbooks.io/tensorflow-kr 입니다.

자세한  사항은 카카오톡 친구 ([GT S골프)](https://tensorflowkorea.wordpress.com/2016/06/01/%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EC%9A%B0-%EB%AC%B8%EC%84%9C-%ED%95%9C%EA%B8%80%ED%99%94/)문의하세요.

감사합니다.

### 번역 및 마크다운 참고사항

#### 번역 범위

현재 텐서플로우는 버전이 빠르게 변하고 있어서 API 문서가 자주 변경됩니다. 따라서 가능하시면 api\_docs 폴더 이외의 다른 문서를 먼저 번역하는 것이 합리적으로 생각됩니다. 각 폴더별 번역 진행 상황은 [@hunkim](https://github.com/hunkim) 님께서 작업해 주신 [progress.md](progress.md) 파일을 참고해 주세요. api\_docs 문서를 제외하고 번역이 완료되면 몇명이서 전체 번역을 다듬으려고 합니다. 이 때 필요한 자원봉사자를 페이스북 텐서플로우 그룹을 통해 신청받으려고 합니다.

#### 버전 관리 안내

텐서플로우 버전이 업데이트 되면서 번역된 문서 간의 버전 차이가 있을 수 있습니다. master 브랜치에는 v0.9의 영문 문서가 들어가 있습니다. 그리고 master 브랜치외에 r0.9, r0.10 두개의 브랜치가 추가 되었습니다. 각각 v0.9, v0.10의 영문 문서가 담겨 있습니다. 두개의 브랜치를 비교하면서 기 번역된 문서를 업데이트 하거나 새로운 번역 문서를 추가할 수 있습니다.

master 브랜치의 번역 문서는 아래와 같이 번역된 버전을 제목 아래에 적어 주십시요. 만약 버전이 적혀있지 않을 경우에는 v0.9로 간주합니다.

```
# 순환 신경망(Recurrent Neural Networks)
(v0.9)
```

#### 아래 pip-installation의 경우처럼 페이지내 이동을 위해서 네임드 앵커 태그가 없이 사용만 되는 경우가 있습니다.

```
*  [Pip 설치](#pip-installation): 이 방식으로 텐서플로우를 설치하거나 업그레이드할 때는
   이 전에 작성했던 파이썬 프로그램에 영향을 미칠 수 있습니다.
```

이럴 때에는 pip-installation 이 가리키는 위치에 네임드 앵커 태그를 넣어주시면 됩니다.

```
...
<a id="pip-installation"></a>
## Pip Installation
...
```
