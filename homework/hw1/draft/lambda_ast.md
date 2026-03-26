```bash

(base) chishan@LAPTOP-7N8BKOTJ:/mnt/d/FudanUniversity/Fdu1/Foundations of Software for Artificial Intelligence/homework/hw1/draft$ python3 -m ast lambda_calculus_intro.py 
Module(
   body=[
      Expr(
         value=Constant(value="\nLambda Calculus Intro (Python Edition)\nLambda 演算入门（Python 版）\n\nThis file uses Python's lambda to simulate core ideas of pure lambda calculus.\n本文件使用 Python 的 lambda 来模拟纯 lambda 演算中的核心思想。\n\nImportant note / 重要说明:\n- Python lambda is NOT the full lambda calculus language.\n  Python 的 lambda 并不等于完整的 lambda 演算语言。\n- But it is a practical bridge for beginners.\n  但它是初学者理解 lambda 演算的实用桥梁。\n")),
      Assign(
         targets=[
            Name(id='IDENTITY', ctx=Store())],
         value=Lambda(
            args=arguments(
               args=[
                  arg(arg='x')]),
            body=Name(id='x', ctx=Load()))),
      Assign(
         targets=[
            Name(id='TRUE', ctx=Store())],
         value=Lambda(
            args=arguments(
               args=[
                  arg(arg='a')]),
            body=Lambda(
               args=arguments(
                  args=[
                     arg(arg='b')]),
               body=Name(id='a', ctx=Load())))),
      Assign(
         targets=[
            Name(id='FALSE', ctx=Store())],
         value=Lambda(
            args=arguments(
               args=[
                  arg(arg='a')]),
            body=Lambda(
               args=arguments(
                  args=[
                     arg(arg='b')]),
               body=Name(id='b', ctx=Load())))),
      Assign(
         targets=[
            Name(id='IF', ctx=Store())],
         value=Lambda(
            args=arguments(
               args=[
                  arg(arg='p')]),
            body=Lambda(
               args=arguments(
                  args=[
                     arg(arg='x')]),
               body=Lambda(
                  args=arguments(
                     args=[
                        arg(arg='y')]),
                  body=Call(
                     func=Call(
                        func=Name(id='p', ctx=Load()),
                        args=[
                           Name(id='x', ctx=Load())]),
                     args=[
                        Name(id='y', ctx=Load())]))))),
      Assign(
         targets=[
            Name(id='ZERO', ctx=Store())],
         value=Lambda(
            args=arguments(
               args=[
                  arg(arg='f')]),
            body=Lambda(
               args=arguments(
                  args=[
                     arg(arg='x')]),
               body=Name(id='x', ctx=Load())))),
      Assign(
         targets=[
            Name(id='ONE', ctx=Store())],
         value=Lambda(
            args=arguments(
               args=[
                  arg(arg='f')]),
            body=Lambda(
               args=arguments(
                  args=[
                     arg(arg='x')]),
               body=Call(
                  func=Name(id='f', ctx=Load()),
                  args=[
                     Name(id='x', ctx=Load())])))),
      Assign(
         targets=[
            Name(id='TWO', ctx=Store())],
         value=Lambda(
            args=arguments(
               args=[
                  arg(arg='f')]),
            body=Lambda(
               args=arguments(
                  args=[
                     arg(arg='x')]),
               body=Call(
                  func=Name(id='f', ctx=Load()),
                  args=[
                     Call(
                        func=Name(id='f', ctx=Load()),
                        args=[
                           Name(id='x', ctx=Load())])])))),
      Assign(
         targets=[
            Name(id='THREE', ctx=Store())],
         value=Lambda(
            args=arguments(
               args=[
                  arg(arg='f')]),
            body=Lambda(
               args=arguments(
                  args=[
                     arg(arg='x')]),
               body=Call(
                  func=Name(id='f', ctx=Load()),
                  args=[
                     Call(
                        func=Name(id='f', ctx=Load()),
                        args=[
                           Call(
                              func=Name(id='f', ctx=Load()),
                              args=[
                                 Name(id='x', ctx=Load())])])])))),
      Assign(
         targets=[
            Name(id='SUCC', ctx=Store())],
         value=Lambda(
            args=arguments(
               args=[
                  arg(arg='n')]),
            body=Lambda(
               args=arguments(
                  args=[
                     arg(arg='f')]),
               body=Lambda(
                  args=arguments(
                     args=[
                        arg(arg='x')]),
                  body=Call(
                     func=Name(id='f', ctx=Load()),
                     args=[
                        Call(
                           func=Call(
                              func=Name(id='n', ctx=Load()),
                              args=[
                                 Name(id='f', ctx=Load())]),
                           args=[
                              Name(id='x', ctx=Load())])]))))),
      Assign(
         targets=[
            Name(id='ADD', ctx=Store())],
         value=Lambda(
            args=arguments(
               args=[
                  arg(arg='m')]),
            body=Lambda(
               args=arguments(
                  args=[
                     arg(arg='n')]),
               body=Lambda(
                  args=arguments(
                     args=[
                        arg(arg='f')]),
                  body=Lambda(
                     args=arguments(
                        args=[
                           arg(arg='x')]),
                     body=Call(
                        func=Call(
                           func=Name(id='m', ctx=Load()),
                           args=[
                              Name(id='f', ctx=Load())]),
                        args=[
                           Call(
                              func=Call(
                                 func=Name(id='n', ctx=Load()),
                                 args=[
                                    Name(id='f', ctx=Load())]),
                              args=[
                                 Name(id='x', ctx=Load())])])))))),
      Assign(
         targets=[
            Name(id='MUL', ctx=Store())],
         value=Lambda(
            args=arguments(
               args=[
                  arg(arg='m')]),
            body=Lambda(
               args=arguments(
                  args=[
                     arg(arg='n')]),
               body=Lambda(
                  args=arguments(
                     args=[
                        arg(arg='f')]),
                  body=Call(
                     func=Name(id='m', ctx=Load()),
                     args=[
                        Call(
                           func=Name(id='n', ctx=Load()),
                           args=[
                              Name(id='f', ctx=Load())])]))))),
      Assign(
         targets=[
            Name(id='to_int', ctx=Store())],
         value=Lambda(
            args=arguments(
               args=[
                  arg(arg='n')]),
            body=Call(
               func=Call(
                  func=Name(id='n', ctx=Load()),
                  args=[
                     Lambda(
                        args=arguments(
                           args=[
                              arg(arg='k')]),
                        body=BinOp(
                           left=Name(id='k', ctx=Load()),
                           op=Add(),
                           right=Constant(value=1)))]),
               args=[
                  Constant(value=0)]))),
      Assign(
         targets=[
            Name(id='to_bool', ctx=Store())],
         value=Lambda(
            args=arguments(
               args=[
                  arg(arg='b')]),
            body=Call(
               func=Call(
                  func=Name(id='b', ctx=Load()),
                  args=[
                     Constant(value=True)]),
               args=[
                  Constant(value=False)]))),
      If(
         test=Compare(
            left=Name(id='__name__', ctx=Load()),
            ops=[
               Eq()],
            comparators=[
               Constant(value='__main__')]),
         body=[
            Expr(
               value=Call(
                  func=Name(id='print', ctx=Load()),
                  args=[
                     Constant(value='=== Lambda Calculus Intro Demo / Lambda 演算入门演示 ===')])),
            Expr(
               value=Call(
                  func=Name(id='print', ctx=Load()),
                  args=[
                     Constant(value='TRUE as Python bool:'),
                     Call(
                        func=Name(id='to_bool', ctx=Load()),
                        args=[
                           Name(id='TRUE', ctx=Load())])])),
            Expr(
               value=Call(
                  func=Name(id='print', ctx=Load()),
                  args=[
                     Constant(value='FALSE as Python bool:'),
                     Call(
                        func=Name(id='to_bool', ctx=Load()),
                        args=[
                           Name(id='FALSE', ctx=Load())])])),
            Assign(
               targets=[
                  Name(id='choose_yes', ctx=Store())],
               value=Call(
                  func=Call(
                     func=Call(
                        func=Name(id='IF', ctx=Load()),
                        args=[
                           Name(id='TRUE', ctx=Load())]),
                     args=[
                        Constant(value='yes')]),
                  args=[
                     Constant(value='no')])),
            Assign(
               targets=[
                  Name(id='choose_no', ctx=Store())],
               value=Call(
                  func=Call(
                     func=Call(
                        func=Name(id='IF', ctx=Load()),
                        args=[
                           Name(id='FALSE', ctx=Load())]),
                     args=[
                        Constant(value='yes')]),
                  args=[
                     Constant(value='no')])),
            Expr(
               value=Call(
                  func=Name(id='print', ctx=Load()),
                  args=[
                     Constant(value='IF(TRUE)(yes)(no):'),
                     Name(id='choose_yes', ctx=Load())])),
            Expr(
               value=Call(
                  func=Name(id='print', ctx=Load()),
                  args=[
                     Constant(value='IF(FALSE)(yes)(no):'),
                     Name(id='choose_no', ctx=Load())])),
            Expr(
               value=Call(
                  func=Name(id='print', ctx=Load()),
                  args=[
                     Constant(value='ZERO:'),
                     Call(
                        func=Name(id='to_int', ctx=Load()),
                        args=[
                           Name(id='ZERO', ctx=Load())])])),
            Expr(
               value=Call(
                  func=Name(id='print', ctx=Load()),
                  args=[
                     Constant(value='ONE:'),
                     Call(
                        func=Name(id='to_int', ctx=Load()),
                        args=[
                           Name(id='ONE', ctx=Load())])])),
            Expr(
               value=Call(
                  func=Name(id='print', ctx=Load()),
                  args=[
                     Constant(value='TWO:'),
                     Call(
                        func=Name(id='to_int', ctx=Load()),
                        args=[
                           Name(id='TWO', ctx=Load())])])),
            Expr(
               value=Call(
                  func=Name(id='print', ctx=Load()),
                  args=[
                     Constant(value='THREE:'),
                     Call(
                        func=Name(id='to_int', ctx=Load()),
                        args=[
                           Name(id='THREE', ctx=Load())])])),
            Assign(
               targets=[
                  Name(id='four', ctx=Store())],
               value=Call(
                  func=Name(id='SUCC', ctx=Load()),
                  args=[
                     Name(id='THREE', ctx=Load())])),
            Assign(
               targets=[
                  Name(id='five', ctx=Store())],
               value=Call(
                  func=Call(
                     func=Name(id='ADD', ctx=Load()),
                     args=[
                        Name(id='TWO', ctx=Load())]),
                  args=[
                     Name(id='THREE', ctx=Load())])),
            Assign(
               targets=[
                  Name(id='six', ctx=Store())],
               value=Call(
                  func=Call(
                     func=Name(id='MUL', ctx=Load()),
                     args=[
                        Name(id='TWO', ctx=Load())]),
                  args=[
                     Name(id='THREE', ctx=Load())])),
            Expr(
               value=Call(
                  func=Name(id='print', ctx=Load()),
                  args=[
                     Constant(value='SUCC(THREE):'),
                     Call(
                        func=Name(id='to_int', ctx=Load()),
                        args=[
                           Name(id='four', ctx=Load())])])),
            Expr(
               value=Call(
                  func=Name(id='print', ctx=Load()),
                  args=[
                     Constant(value='ADD(TWO)(THREE):'),
                     Call(
                        func=Name(id='to_int', ctx=Load()),
                        args=[
                           Name(id='five', ctx=Load())])])),
            Expr(
               value=Call(
                  func=Name(id='print', ctx=Load()),
                  args=[
                     Constant(value='MUL(TWO)(THREE):'),
                     Call(
                        func=Name(id='to_int', ctx=Load()),
                        args=[
                           Name(id='six', ctx=Load())])])),
            Expr(
               value=Call(
                  func=Name(id='print', ctx=Load()),
                  args=[
                     Constant(value='Takeaway: numerals are repeated function application.')]))])])
```