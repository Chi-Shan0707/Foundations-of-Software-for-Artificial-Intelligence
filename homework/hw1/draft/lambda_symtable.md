```bash
(base) chishan@LAPTOP-7N8BKOTJ:/mnt/d/FudanUniversity/Fdu1/Foundations of Software for Artificial Intelligence/homework/hw1/draft$ python3 -m symtable lambda_calculus_intro.py 
symbol table for module from file 'lambda_calculus_intro.py':
    local symbol 'IDENTITY': def_local
    local symbol 'TRUE': use, def_local
    local symbol 'FALSE': use, def_local
    local symbol 'IF': use, def_local
    local symbol 'ZERO': use, def_local
    local symbol 'ONE': use, def_local
    local symbol 'TWO': use, def_local
    local symbol 'THREE': use, def_local
    local symbol 'SUCC': use, def_local
    local symbol 'ADD': use, def_local
    local symbol 'MUL': use, def_local
    local symbol 'to_int': use, def_local
    local symbol 'to_bool': use, def_local
    global_implicit symbol '__name__': use
    global_implicit symbol 'print': use
    local symbol 'choose_yes': use, def_local
    local symbol 'choose_no': use, def_local
    local symbol 'four': use, def_local
    local symbol 'five': use, def_local
    local symbol 'six': use, def_local

    symbol table for function 'lambda':
        local symbol 'x': use, def_param

    symbol table for function 'lambda':
        cell symbol 'a': def_param

        symbol table for nested function 'lambda':
            local symbol 'b': def_param
            free symbol 'a': use

    symbol table for function 'lambda':
        local symbol 'a': def_param

        symbol table for nested function 'lambda':
            local symbol 'b': use, def_param

    symbol table for function 'lambda':
        cell symbol 'p': def_param

        symbol table for nested function 'lambda':
            cell symbol 'x': def_param
            free symbol 'p': 

            symbol table for nested function 'lambda':
                local symbol 'y': use, def_param
                free symbol 'p': use
                free symbol 'x': use

    symbol table for function 'lambda':
        local symbol 'f': def_param

        symbol table for nested function 'lambda':
            local symbol 'x': use, def_param

    symbol table for function 'lambda':
        cell symbol 'f': def_param

        symbol table for nested function 'lambda':
            local symbol 'x': use, def_param
            free symbol 'f': use

    symbol table for function 'lambda':
        cell symbol 'f': def_param

        symbol table for nested function 'lambda':
            local symbol 'x': use, def_param
            free symbol 'f': use

    symbol table for function 'lambda':
        cell symbol 'f': def_param

        symbol table for nested function 'lambda':
            local symbol 'x': use, def_param
            free symbol 'f': use

    symbol table for function 'lambda':
        cell symbol 'n': def_param

        symbol table for nested function 'lambda':
            cell symbol 'f': def_param
            free symbol 'n': 

            symbol table for nested function 'lambda':
                local symbol 'x': use, def_param
                free symbol 'f': use
                free symbol 'n': use

    symbol table for function 'lambda':
        cell symbol 'm': def_param

        symbol table for nested function 'lambda':
            cell symbol 'n': def_param
            free symbol 'm': 

            symbol table for nested function 'lambda':
                cell symbol 'f': def_param
                free symbol 'n': 
                free symbol 'm': 

                symbol table for nested function 'lambda':
                    local symbol 'x': use, def_param
                    free symbol 'm': use
                    free symbol 'f': use
                    free symbol 'n': use

    symbol table for function 'lambda':
        cell symbol 'm': def_param

        symbol table for nested function 'lambda':
            cell symbol 'n': def_param
            free symbol 'm': 

            symbol table for nested function 'lambda':
                local symbol 'f': use, def_param
                free symbol 'm': use
                free symbol 'n': use

    symbol table for function 'lambda':
        local symbol 'n': use, def_param

        symbol table for nested function 'lambda':
            local symbol 'k': use, def_param

    symbol table for function 'lambda':
        local symbol 'b': use, def_param
```