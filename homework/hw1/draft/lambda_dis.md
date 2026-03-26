```bash


 python3 -m dis lambda_calculus_intro.py 
  0           RESUME                   0

  1           LOAD_CONST               0 ("\nLambda Calculus Intro (Python Edition)\nLambda 演算入门（Python 版）\n\nThis file uses Python's lambda to simulate core ideas of pure lambda calculus.\n本文件使用 Python 的 lambda 来模拟纯 lambda 演算中的核心思想。\n\nImportant note / 重要说明:\n- Python lambda is NOT the full lambda calculus language.\n  Python 的 lambda 并不等于完整的 lambda 演算语言。\n- But it is a practical bridge for beginners.\n  但它是初学者理解 lambda 演算的实用桥梁。\n")
              STORE_NAME               0 (__doc__)

 23           LOAD_CONST               1 (<code object <lambda> at 0x703cc0c812e0, file "lambda_calculus_intro.py", line 23>)
              MAKE_FUNCTION
              STORE_NAME               1 (IDENTITY)

 33           LOAD_CONST               2 (<code object <lambda> at 0x703cc0f86a30, file "lambda_calculus_intro.py", line 33>)
              MAKE_FUNCTION
              STORE_NAME               2 (TRUE)

 34           LOAD_CONST               3 (<code object <lambda> at 0x703cc0c81550, file "lambda_calculus_intro.py", line 34>)
              MAKE_FUNCTION
              STORE_NAME               3 (FALSE)

 38           LOAD_CONST               4 (<code object <lambda> at 0x703cc0e325d0, file "lambda_calculus_intro.py", line 38>)
              MAKE_FUNCTION
              STORE_NAME               4 (IF)

 52           LOAD_CONST               5 (<code object <lambda> at 0x703cc0c816f0, file "lambda_calculus_intro.py", line 52>)
              MAKE_FUNCTION
              STORE_NAME               5 (ZERO)

 53           LOAD_CONST               6 (<code object <lambda> at 0x703cc0e32410, file "lambda_calculus_intro.py", line 53>)
              MAKE_FUNCTION
              STORE_NAME               6 (ONE)

 54           LOAD_CONST               7 (<code object <lambda> at 0x703cc0e326b0, file "lambda_calculus_intro.py", line 54>)
              MAKE_FUNCTION
              STORE_NAME               7 (TWO)

 55           LOAD_CONST               8 (<code object <lambda> at 0x703cc0e32870, file "lambda_calculus_intro.py", line 55>)
              MAKE_FUNCTION
              STORE_NAME               8 (THREE)

 60           LOAD_CONST               9 (<code object <lambda> at 0x703cc0e334b0, file "lambda_calculus_intro.py", line 60>)
              MAKE_FUNCTION
              STORE_NAME               9 (SUCC)

 65           LOAD_CONST              10 (<code object <lambda> at 0x703cc0e33590, file "lambda_calculus_intro.py", line 65>)
              MAKE_FUNCTION
              STORE_NAME              10 (ADD)

 70           LOAD_CONST              11 (<code object <lambda> at 0x703cc0e33910, file "lambda_calculus_intro.py", line 70>)
              MAKE_FUNCTION
              STORE_NAME              11 (MUL)

 78           LOAD_CONST              12 (<code object <lambda> at 0x703cc0e37a50, file "lambda_calculus_intro.py", line 78>)
              MAKE_FUNCTION
              STORE_NAME              12 (to_int)

 82           LOAD_CONST              13 (<code object <lambda> at 0x703cc0e37b40, file "lambda_calculus_intro.py", line 82>)
              MAKE_FUNCTION
              STORE_NAME              13 (to_bool)

 88           LOAD_NAME               14 (__name__)
              LOAD_CONST              14 ('__main__')
              COMPARE_OP              88 (bool(==))
              POP_JUMP_IF_FALSE      246 (to L1)

 89           LOAD_NAME               15 (print)
              PUSH_NULL
              LOAD_CONST              15 ('=== Lambda Calculus Intro Demo / Lambda 演算入门演示 ===')
              CALL                     1
              POP_TOP

 92           LOAD_NAME               15 (print)
              PUSH_NULL
              LOAD_CONST              16 ('TRUE as Python bool:')
              LOAD_NAME               13 (to_bool)
              PUSH_NULL
              LOAD_NAME                2 (TRUE)
              CALL                     1
              CALL                     2
              POP_TOP

 93           LOAD_NAME               15 (print)
              PUSH_NULL
              LOAD_CONST              17 ('FALSE as Python bool:')
              LOAD_NAME               13 (to_bool)
              PUSH_NULL
              LOAD_NAME                3 (FALSE)
              CALL                     1
              CALL                     2
              POP_TOP

 97           LOAD_NAME                4 (IF)
              PUSH_NULL
              LOAD_NAME                2 (TRUE)
              CALL                     1
              PUSH_NULL
              LOAD_CONST              18 ('yes')
              CALL                     1
              PUSH_NULL
              LOAD_CONST              19 ('no')
              CALL                     1
              STORE_NAME              16 (choose_yes)

 98           LOAD_NAME                4 (IF)
              PUSH_NULL
              LOAD_NAME                3 (FALSE)
              CALL                     1
              PUSH_NULL
              LOAD_CONST              18 ('yes')
              CALL                     1
              PUSH_NULL
              LOAD_CONST              19 ('no')
              CALL                     1
              STORE_NAME              17 (choose_no)

 99           LOAD_NAME               15 (print)
              PUSH_NULL
              LOAD_CONST              20 ('IF(TRUE)(yes)(no):')
              LOAD_NAME               16 (choose_yes)
              CALL                     2
              POP_TOP

100           LOAD_NAME               15 (print)
              PUSH_NULL
              LOAD_CONST              21 ('IF(FALSE)(yes)(no):')
              LOAD_NAME               17 (choose_no)
              CALL                     2
              POP_TOP

103           LOAD_NAME               15 (print)
              PUSH_NULL
              LOAD_CONST              22 ('ZERO:')
              LOAD_NAME               12 (to_int)
              PUSH_NULL
              LOAD_NAME                5 (ZERO)
              CALL                     1
              CALL                     2
              POP_TOP

104           LOAD_NAME               15 (print)
              PUSH_NULL
              LOAD_CONST              23 ('ONE:')
              LOAD_NAME               12 (to_int)
              PUSH_NULL
              LOAD_NAME                6 (ONE)
              CALL                     1
              CALL                     2
              POP_TOP

105           LOAD_NAME               15 (print)
              PUSH_NULL
              LOAD_CONST              24 ('TWO:')
              LOAD_NAME               12 (to_int)
              PUSH_NULL
              LOAD_NAME                7 (TWO)
              CALL                     1
              CALL                     2
              POP_TOP

106           LOAD_NAME               15 (print)
              PUSH_NULL
              LOAD_CONST              25 ('THREE:')
              LOAD_NAME               12 (to_int)
              PUSH_NULL
              LOAD_NAME                8 (THREE)
              CALL                     1
              CALL                     2
              POP_TOP

108           LOAD_NAME                9 (SUCC)
              PUSH_NULL
              LOAD_NAME                8 (THREE)
              CALL                     1
              STORE_NAME              18 (four)

109           LOAD_NAME               10 (ADD)
              PUSH_NULL
              LOAD_NAME                7 (TWO)
              CALL                     1
              PUSH_NULL
              LOAD_NAME                8 (THREE)
              CALL                     1
              STORE_NAME              19 (five)

110           LOAD_NAME               11 (MUL)
              PUSH_NULL
              LOAD_NAME                7 (TWO)
              CALL                     1
              PUSH_NULL
              LOAD_NAME                8 (THREE)
              CALL                     1
              STORE_NAME              20 (six)

112           LOAD_NAME               15 (print)
              PUSH_NULL
              LOAD_CONST              26 ('SUCC(THREE):')
              LOAD_NAME               12 (to_int)
              PUSH_NULL
              LOAD_NAME               18 (four)
              CALL                     1
              CALL                     2
              POP_TOP

113           LOAD_NAME               15 (print)
              PUSH_NULL
              LOAD_CONST              27 ('ADD(TWO)(THREE):')
              LOAD_NAME               12 (to_int)
              PUSH_NULL
              LOAD_NAME               19 (five)
              CALL                     1
              CALL                     2
              POP_TOP

114           LOAD_NAME               15 (print)
              PUSH_NULL
              LOAD_CONST              28 ('MUL(TWO)(THREE):')
              LOAD_NAME               12 (to_int)
              PUSH_NULL
              LOAD_NAME               20 (six)
              CALL                     1
              CALL                     2
              POP_TOP

119           LOAD_NAME               15 (print)
              PUSH_NULL
              LOAD_CONST              29 ('Takeaway: numerals are repeated function application.')
              CALL                     1
              POP_TOP
              RETURN_CONST            30 (None)

 88   L1:     RETURN_CONST            30 (None)

Disassembly of <code object <lambda> at 0x703cc0c812e0, file "lambda_calculus_intro.py", line 23>:
 23           RESUME                   0
              LOAD_FAST                0 (x)
              RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0f86a30, file "lambda_calculus_intro.py", line 33>:
  --           MAKE_CELL                0 (a)

  33           RESUME                   0
               LOAD_FAST                0 (a)
               BUILD_TUPLE              1
               LOAD_CONST               1 (<code object <lambda> at 0x703cc0c813b0, file "lambda_calculus_intro.py", line 33>)
               MAKE_FUNCTION
               SET_FUNCTION_ATTRIBUTE   8 (closure)
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0c813b0, file "lambda_calculus_intro.py", line 33>:
  --           COPY_FREE_VARS           1

  33           RESUME                   0
               LOAD_DEREF               1 (a)
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0c81550, file "lambda_calculus_intro.py", line 34>:
 34           RESUME                   0
              LOAD_CONST               1 (<code object <lambda> at 0x703cc0c81480, file "lambda_calculus_intro.py", line 34>)
              MAKE_FUNCTION
              RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0c81480, file "lambda_calculus_intro.py", line 34>:
 34           RESUME                   0
              LOAD_FAST                0 (b)
              RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e325d0, file "lambda_calculus_intro.py", line 38>:
  --           MAKE_CELL                0 (p)

  38           RESUME                   0
               LOAD_FAST                0 (p)
               BUILD_TUPLE              1
               LOAD_CONST               1 (<code object <lambda> at 0x703cc0e32090, file "lambda_calculus_intro.py", line 38>)
               MAKE_FUNCTION
               SET_FUNCTION_ATTRIBUTE   8 (closure)
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e32090, file "lambda_calculus_intro.py", line 38>:
  --           COPY_FREE_VARS           1
               MAKE_CELL                0 (x)

  38           RESUME                   0
               LOAD_FAST                1 (p)
               LOAD_FAST                0 (x)
               BUILD_TUPLE              2
               LOAD_CONST               1 (<code object <lambda> at 0x703cc0e374b0, file "lambda_calculus_intro.py", line 38>)
               MAKE_FUNCTION
               SET_FUNCTION_ATTRIBUTE   8 (closure)
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e374b0, file "lambda_calculus_intro.py", line 38>:
  --           COPY_FREE_VARS           2

  38           RESUME                   0
               LOAD_DEREF               1 (p)
               PUSH_NULL
               LOAD_DEREF               2 (x)
               CALL                     1
               PUSH_NULL
               LOAD_FAST                0 (y)
               CALL                     1
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0c816f0, file "lambda_calculus_intro.py", line 52>:
 52           RESUME                   0
              LOAD_CONST               1 (<code object <lambda> at 0x703cc0c81620, file "lambda_calculus_intro.py", line 52>)
              MAKE_FUNCTION
              RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0c81620, file "lambda_calculus_intro.py", line 52>:
 52           RESUME                   0
              LOAD_FAST                0 (x)
              RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e32410, file "lambda_calculus_intro.py", line 53>:
  --           MAKE_CELL                0 (f)

  53           RESUME                   0
               LOAD_FAST                0 (f)
               BUILD_TUPLE              1
               LOAD_CONST               1 (<code object <lambda> at 0x703cc0e332f0, file "lambda_calculus_intro.py", line 53>)
               MAKE_FUNCTION
               SET_FUNCTION_ATTRIBUTE   8 (closure)
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e332f0, file "lambda_calculus_intro.py", line 53>:
  --           COPY_FREE_VARS           1

  53           RESUME                   0
               LOAD_DEREF               1 (f)
               PUSH_NULL
               LOAD_FAST                0 (x)
               CALL                     1
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e326b0, file "lambda_calculus_intro.py", line 54>:
  --           MAKE_CELL                0 (f)

  54           RESUME                   0
               LOAD_FAST                0 (f)
               BUILD_TUPLE              1
               LOAD_CONST               1 (<code object <lambda> at 0x703cc0e37870, file "lambda_calculus_intro.py", line 54>)
               MAKE_FUNCTION
               SET_FUNCTION_ATTRIBUTE   8 (closure)
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e37870, file "lambda_calculus_intro.py", line 54>:
  --           COPY_FREE_VARS           1

  54           RESUME                   0
               LOAD_DEREF               1 (f)
               PUSH_NULL
               LOAD_DEREF               1 (f)
               PUSH_NULL
               LOAD_FAST                0 (x)
               CALL                     1
               CALL                     1
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e32870, file "lambda_calculus_intro.py", line 55>:
  --           MAKE_CELL                0 (f)

  55           RESUME                   0
               LOAD_FAST                0 (f)
               BUILD_TUPLE              1
               LOAD_CONST               1 (<code object <lambda> at 0x703cc0e60d30, file "lambda_calculus_intro.py", line 55>)
               MAKE_FUNCTION
               SET_FUNCTION_ATTRIBUTE   8 (closure)
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e60d30, file "lambda_calculus_intro.py", line 55>:
  --           COPY_FREE_VARS           1

  55           RESUME                   0
               LOAD_DEREF               1 (f)
               PUSH_NULL
               LOAD_DEREF               1 (f)
               PUSH_NULL
               LOAD_DEREF               1 (f)
               PUSH_NULL
               LOAD_FAST                0 (x)
               CALL                     1
               CALL                     1
               CALL                     1
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e334b0, file "lambda_calculus_intro.py", line 60>:
  --           MAKE_CELL                0 (n)

  60           RESUME                   0
               LOAD_FAST                0 (n)
               BUILD_TUPLE              1
               LOAD_CONST               1 (<code object <lambda> at 0x703cc0e324f0, file "lambda_calculus_intro.py", line 60>)
               MAKE_FUNCTION
               SET_FUNCTION_ATTRIBUTE   8 (closure)
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e324f0, file "lambda_calculus_intro.py", line 60>:
  --           COPY_FREE_VARS           1
               MAKE_CELL                0 (f)

  60           RESUME                   0
               LOAD_FAST                0 (f)
               LOAD_FAST                1 (n)
               BUILD_TUPLE              2
               LOAD_CONST               1 (<code object <lambda> at 0x703cc0e61630, file "lambda_calculus_intro.py", line 60>)
               MAKE_FUNCTION
               SET_FUNCTION_ATTRIBUTE   8 (closure)
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e61630, file "lambda_calculus_intro.py", line 60>:
  --           COPY_FREE_VARS           2

  60           RESUME                   0
               LOAD_DEREF               1 (f)
               PUSH_NULL
               LOAD_DEREF               2 (n)
               PUSH_NULL
               LOAD_DEREF               1 (f)
               CALL                     1
               PUSH_NULL
               LOAD_FAST                0 (x)
               CALL                     1
               CALL                     1
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e33590, file "lambda_calculus_intro.py", line 65>:
  --           MAKE_CELL                0 (m)

  65           RESUME                   0
               LOAD_FAST                0 (m)
               BUILD_TUPLE              1
               LOAD_CONST               1 (<code object <lambda> at 0x703cc0e333d0, file "lambda_calculus_intro.py", line 65>)
               MAKE_FUNCTION
               SET_FUNCTION_ATTRIBUTE   8 (closure)
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e333d0, file "lambda_calculus_intro.py", line 65>:
  --           COPY_FREE_VARS           1
               MAKE_CELL                0 (n)

  65           RESUME                   0
               LOAD_FAST                1 (m)
               LOAD_FAST                0 (n)
               BUILD_TUPLE              2
               LOAD_CONST               1 (<code object <lambda> at 0x703cc0e339f0, file "lambda_calculus_intro.py", line 65>)
               MAKE_FUNCTION
               SET_FUNCTION_ATTRIBUTE   8 (closure)
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e339f0, file "lambda_calculus_intro.py", line 65>:
  --           COPY_FREE_VARS           2
               MAKE_CELL                0 (f)

  65           RESUME                   0
               LOAD_FAST                0 (f)
               LOAD_FAST                1 (m)
               LOAD_FAST                2 (n)
               BUILD_TUPLE              3
               LOAD_CONST               1 (<code object <lambda> at 0x703cc0e61730, file "lambda_calculus_intro.py", line 65>)
               MAKE_FUNCTION
               SET_FUNCTION_ATTRIBUTE   8 (closure)
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e61730, file "lambda_calculus_intro.py", line 65>:
  --           COPY_FREE_VARS           3

  65           RESUME                   0
               LOAD_DEREF               2 (m)
               PUSH_NULL
               LOAD_DEREF               1 (f)
               CALL                     1
               PUSH_NULL
               LOAD_DEREF               3 (n)
               PUSH_NULL
               LOAD_DEREF               1 (f)
               CALL                     1
               PUSH_NULL
               LOAD_FAST                0 (x)
               CALL                     1
               CALL                     1
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e33910, file "lambda_calculus_intro.py", line 70>:
  --           MAKE_CELL                0 (m)

  70           RESUME                   0
               LOAD_FAST                0 (m)
               BUILD_TUPLE              1
               LOAD_CONST               1 (<code object <lambda> at 0x703cc0e33bb0, file "lambda_calculus_intro.py", line 70>)
               MAKE_FUNCTION
               SET_FUNCTION_ATTRIBUTE   8 (closure)
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e33bb0, file "lambda_calculus_intro.py", line 70>:
  --           COPY_FREE_VARS           1
               MAKE_CELL                0 (n)

  70           RESUME                   0
               LOAD_FAST                1 (m)
               LOAD_FAST                0 (n)
               BUILD_TUPLE              2
               LOAD_CONST               1 (<code object <lambda> at 0x703cc0e37960, file "lambda_calculus_intro.py", line 70>)
               MAKE_FUNCTION
               SET_FUNCTION_ATTRIBUTE   8 (closure)
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e37960, file "lambda_calculus_intro.py", line 70>:
  --           COPY_FREE_VARS           2

  70           RESUME                   0
               LOAD_DEREF               1 (m)
               PUSH_NULL
               LOAD_DEREF               2 (n)
               PUSH_NULL
               LOAD_FAST                0 (f)
               CALL                     1
               CALL                     1
               RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e37a50, file "lambda_calculus_intro.py", line 78>:
 78           RESUME                   0
              LOAD_FAST                0 (n)
              PUSH_NULL
              LOAD_CONST               1 (<code object <lambda> at 0x703cc0e33750, file "lambda_calculus_intro.py", line 78>)
              MAKE_FUNCTION
              CALL                     1
              PUSH_NULL
              LOAD_CONST               2 (0)
              CALL                     1
              RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e33750, file "lambda_calculus_intro.py", line 78>:
 78           RESUME                   0
              LOAD_FAST                0 (k)
              LOAD_CONST               1 (1)
              BINARY_OP                0 (+)
              RETURN_VALUE

Disassembly of <code object <lambda> at 0x703cc0e37b40, file "lambda_calculus_intro.py", line 82>:
 82           RESUME                   0
              LOAD_FAST                0 (b)
              PUSH_NULL
              LOAD_CONST               1 (True)
              CALL                     1
              PUSH_NULL
              LOAD_CONST               2 (False)
              CALL                     1
              RETURN_VALUE

```