---
layout:
  title:
    visible: true
  description:
    visible: true
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: false
---

# 🪛 A/S 메뉴얼



```php
// Code Start
<?php
// plugins/myplugin/src/ToggleMenus/ToggleItem.php
namespace MyPlugin\ToggleMenus;

class ToggleItem extends AbstractToggleMenu
{
    public function getText()
    {
      return 'string';
    }

    public function getType()
    {
      // implement code

      return static::MENUTYPE_RAW
    }

    public function getAction()
    {
      return 'do_something()';
    }

    public function getScript()
    {
      return 'js_file_directory';
    }
}

```



```php
// Some code
```
