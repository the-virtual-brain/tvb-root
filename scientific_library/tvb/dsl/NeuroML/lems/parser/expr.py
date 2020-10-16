"""
Expression parser

@author: Gautham Ganapathy
@organization: LEMS (http://neuroml.org/lems/, https://github.com/organizations/LEMS)
@contact: gautham@lisphacker.org

MAvdVlag: added pow(f,l) function for c translations and ** for python power function. Removed H as a known function
"""

from ..base.base import LEMSBase
from ..base.stack import Stack

known_functions = ['exp', 'log', 'sqrt', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'abs', 'ceil', 'factorial', 'random', 'pow', 'powf', 'powl']

class ExprNode(LEMSBase):
    """
    Base class for a node in the expression parse tree.
    """

    OP = 1
    VALUE = 2
    FUNC1 = 3

    def __init__(self, type):
        """
        Constructor.

        @param type: Node type
        @type type: enum(ExprNode.OP, ExprNode.VALUE)
        """

        self.type = type
        """ Node type.
        @type: enum(ExprNode.OP, ExprNode.VALUE) """

class ValueNode(ExprNode):
    """
    Value node in an expression parse tree. This will always be a leaf node.
    """

    def __init__(self, value):
        """
        Constructor.

        @param value: Value to be stored in this node.
        @type value: string
        """

        ExprNode.__init__(self, ExprNode.VALUE)
        self.value = value
        """ Value to be stored in this node.
        @type: string """
        
    def clean_up(self):
        """
        To make sure an integer is returned as a float. No division by integers!!
        """
        try:
            return str(float(self.value)) 
        except ValueError:
            return self.value
                

    def __str__(self):
        """
        Generates a string representation of this node.
        """
        return "{" + self.clean_up() + "}"
    
    def __repr__(self):
        return self.__str__()
    
    def to_python_expr(self):
        return self.clean_up()

class OpNode(ExprNode):
    """
    Operation node in an expression parse tree. This will always be a
    non-leaf node.
    """

    def __init__(self, op, left, right):
        """
        Constructor.

        @param op: Operation to be stored in this node.
        @type op: string

        @param left: Left operand.
        @type left: lems.parser.expr.ExprNode

        @param right: Right operand.
        @type right: lems.parser.expr.ExprNode
        """

        ExprNode.__init__(self, ExprNode.OP)

        self.op = op
        """ Operation stored in this node.
        @type: string """

        self.left = left
        """ Left operand.
        @type: lems.parser.expr.ExprNode """

        self.right = right
        """ Right operand.
        @type: lems.parser.expr.ExprNode """

    def __str__(self):
        """
        Generates a string representation of this node.
        """

        return '({0} {1} {2})'.format(self.op,
                                      str(self.left),
                                      str(self.right))
                                      
    def __repr__(self):
        return self.__str__()
    
    def to_python_expr(self):
        return '({0} {1} {2})'.format(self.left.to_python_expr(),
                                      self.op,
                                      self.right.to_python_expr())

class Func1Node(ExprNode):
    """
    Unary function node in an expression parse tree. This will always be a
    non-leaf node.
    """

    def __init__(self, func, param):
        """
        Constructor.

        @param func: Function to be stored in this node.
        @type func: string

        @param param: Parameter.
        @type param: lems.parser.expr.ExprNode
        """

        ExprNode.__init__(self, ExprNode.FUNC1)

        self.func = func
        """ Funcion stored in this node.
        @type: string """

        self.param = param
        """ Parameter.
        @type: lems.parser.expr.ExprNode """

    def __str__(self):
        """
        Generates a string representation of this node.
        """

        return '({0} {1})'.format(self.func, str(self.param))
    
    def __repr__(self):
        return self.__str__()
    
    def to_python_expr(self):
        return '({0}({1}))'.format(self.func, self.param.to_python_expr())


class ExprParser(LEMSBase):
    """
    Parser class for parsing an expression and generating a parse tree.
    """

    debug = False

    op_priority = {
        '$':-5,
        'func':8,
        '**':6,
        '+':5,
        '-':5,
        '*':6,
        '/':6,
        '^':7,
        '~':8,
        'exp':8,
        'numpy.exp': 8,
        '.and.':1,
        '.or.':1,
        '.gt.':2,
        '.ge.':2,
        '.geq.':2,
        '.lt.':2,
        '.le.':2,
        '.eq.':2,
        '.neq.':2,
        '.ne.':2}  # .neq. is preferred!

    depth = 0

    """ Dictionary mapping operators to their priorities.
    @type: dict(string -> Integer) """

    def __init__(self, parse_string):
        """
        Constructor.

        @param parse_string: Expression to be parsed.
        @type parse_string: string
        """

        self.parse_string = parse_string
        """ Expression to be parsed.
        @type: string """

        self.token_list = None
        """ List of tokens from the expression to be parsed.
        @type: list(string) """

    def is_op(self, str):
        """
        Checks if a token string contains an operator.

        @param str: Token string to be checked.
        @type str: string

        @return: True if the token string contains an operator.
        @rtype: Boolean
        """

        return str in self.op_priority

    def is_func(self, str):
        """
        Checks if a token string contains a function.

        @param str: Token string to be checked.
        @type str: string

        @return: True if the token string contains a function.
        @rtype: Boolean
        """

        return str in known_functions

    def is_sym(self, str):
        """
        Checks if a token string contains a symbol.

        @param str: Token string to be checked.
        @type str: string

        @return: True if the token string contains a symbol.
        @rtype: Boolean

        MV: Added the part to recognize ** for power
        """

        return str in ['+', '-', '~', '*', '/', '^', '(', ')']

    def priority(self, op):
        if self.is_op(op):
            return self.op_priority[op]
        elif self.is_func(op):
            return self.op_priority['func']
        else:
            return self.op_priority['$']

    def tokenize(self):
        """
        Tokenizes the string stored in the parser object into a list
        of tokens.
        """

        powerflag = 0
        self.token_list = []
        ps = self.parse_string.strip()

        i = 0
        last_token = None

        while i < len(ps) and ps[i].isspace():
            i += 1

        while i < len(ps):
            token = ''

            if ps[i].isalpha():
                while i < len(ps) and (ps[i].isalnum() or ps[i] == '_'):
                    token += ps[i]
                    i += 1
            elif ps[i].isdigit():
                while i < len(ps) and (ps[i].isdigit() or
                                       ps[i] == '.' or
                                       ps[i] == 'e' or
                                       ps[i] == 'E' or
                                       (ps[i] == '+' and (ps[i-1] == 'e' or ps[i-1] == 'E')) or
                                       (ps[i] == '-' and (ps[i-1] == 'e' or ps[i-1] == 'E'))):
                    token += ps[i]
                    i += 1
            elif ps[i] == '.':
                if ps[i+1].isdigit():
                    while i < len(ps) and (ps[i].isdigit() or ps[i] == '.'):
                        token += ps[i]
                        i += 1
                else:
                    while i < len(ps) and (ps[i].isalpha() or ps[i] == '.'):
                        token += ps[i]
                        i += 1
            elif ps[i] == '*':
                while i < len(ps) and ps[i] == '*':# and j < 2:
                    token += ps[i]
                    i += 1
            else:
                token += ps[i]
                i += 1

            if token == '-' and \
               (last_token == None or last_token == '(' or self.is_op(last_token)):
                token = '~'

            self.token_list += [token]
            last_token = token

            while i < len(ps) and ps[i].isspace():
                i += 1
                
    def make_op_node(self, op, right):
        if self.is_func(op):
            return Func1Node(op, right)
        elif op == '~':
            return OpNode('-', ValueNode('0'), right)
        else:
            left = self.val_stack.pop()
            if left == '$':
                left = self.node_stack.pop()
            else:
                left = ValueNode(left)

            return OpNode(op, left, right)

    def cleanup_stacks(self):
        right = self.val_stack.pop()
        if right == '$':
            right = self.node_stack.pop()
        else:
            right = ValueNode(right)
            
        if self.debug: print('- Cleanup > right: %s'% right)
        
        while self.op_stack.top() != '$':
            if self.debug: print('5> op stack: %s, val stack: %s, node stack: %s'% ( self.op_stack, self.val_stack, self.node_stack))
            op = self.op_stack.pop()

            right = self.make_op_node(op, right)

            if self.debug: print('6> op stack: %s, val stack: %s, node stack: %s'% ( self.op_stack, self.val_stack, self.node_stack))
            if self.debug: print('7> %s'% right)
            #if self.debug: print(''

        return right

    def parse_token_list_rec(self, min_precedence):
        """
        Parses a tokenized arithmetic expression into a parse tree. It calls
        itself recursively to handle bracketed subexpressions.

        @return: Returns a token string.
        @rtype: lems.parser.expr.ExprNode

        @attention: Does not handle unary minuses at the moment. Needs to be
        fixed.
        """

        exit_loop = False

        ExprParser.depth = ExprParser.depth + 1
        if self.debug: print('>>>>> Depth: %i'% ExprParser.depth)

        precedence = min_precedence

        while self.token_list:
            token = self.token_list[0]
            la = self.token_list[1] if len(self.token_list) > 1 else None

            if self.debug: print('0> %s'% self.token_list)
            if self.debug: print('1> Token: %s, next: %s, op stack: %s, val stack: %s, node stack: %s'% (token, la, self.op_stack, self.val_stack, self.node_stack))

            self.token_list = self.token_list[1:]
            
            close_bracket = False
            
            if token == '(':
                np = ExprParser('')
                np.token_list = self.token_list

                nexp = np.parse2()

                self.node_stack.push(nexp)
                self.val_stack.push('$')

                self.token_list = np.token_list
                if self.debug: print('>>> Tokens left: %s'%self.token_list)
                close_bracket = True
            elif token == ')':
                break
            elif self.is_func(token):
                self.op_stack.push(token)
            elif self.is_op(token):
                stack_top = self.op_stack.top()
                if self.debug: print('OP Token: %s (prior: %i), top: %s (prior: %i)'% (token, self.priority(token), stack_top, self.priority(stack_top)))
                if self.priority(token) < self.priority(stack_top):
                    if self.debug: print('  Priority of %s is less than %s'%(token, stack_top))
                    self.node_stack.push(self.cleanup_stacks())
                    self.val_stack.push('$')
                else:
                    if self.debug: print('  Priority of %s is greater than %s'%(token, stack_top))
                

                self.op_stack.push(token)
            else:
                if self.debug: print('Not a bracket func or op...')
                if la == '(':
                    raise Exception("Error parsing expression: %s\nToken: %s is placed like a function but is not recognised!\nKnown functions: %s"%(self.parse_string, token, known_functions))
                stack_top = self.op_stack.top()
                if stack_top == '$':
                    if self.debug: print("option a")
                    self.node_stack.push(ValueNode(token))
                    self.val_stack.push('$')
                else:
                    if (self.is_op(la) and
                        self.priority(stack_top) < self.priority(la)):
                        if self.debug: print("option b")

                        self.node_stack.push(ValueNode(token))
                        self.val_stack.push('$')
                    else:
                        if self.debug: print("option c, nodes: %s"% self.node_stack)
                        op = self.op_stack.pop()

                        right = ValueNode(token)
                        op_node = self.make_op_node(op,right)

                        self.node_stack.push(op_node)
                        self.val_stack.push('$')
                        
            if close_bracket:
                stack_top = self.op_stack.top()
                if self.debug: print("+ Closing bracket, op stack: %s, node stack: %s la: %s"%(self.op_stack, self.node_stack, la))
                if self.debug: print('>>> Tokens left: %s'%self.token_list)
                
                if stack_top == '$':
                    if self.debug: print("+ option a")
                    '''
                    self.node_stack.push(ValueNode(token))
                    self.val_stack.push('$')'''
                else:
                    la = self.token_list[0] if len(self.token_list) > 1 else None
                    if (self.is_op(la) and self.priority(stack_top) < self.priority(la)):
                        if self.debug: print("+ option b")
                        #self.node_stack.push(ValueNode(token))
                        #self.val_stack.push('$')
                    else:
                        if self.debug: print("+ option c, nodes: %s"% self.node_stack)
                        if self.debug: print('35> op stack: %s, val stack: %s, node stack: %s'% ( self.op_stack, self.val_stack, self.node_stack))
                        right = self.node_stack.pop()
                        op = self.op_stack.pop()
                        op_node = self.make_op_node(stack_top,right)
                        if self.debug: print("Made op node: %s, right: %s"%(op_node, right))

                        self.node_stack.push(op_node)
                        self.val_stack.push('$')
                        if self.debug: print('36> op stack: %s, val stack: %s, node stack: %s'% ( self.op_stack, self.val_stack, self.node_stack))
                        
            

            if self.debug: print('2> Token: %s, next: %s, op stack: %s, val stack: %s, node stack: %s'% (token, la, self.op_stack, self.val_stack, self.node_stack))
            if self.debug: print('')

        if self.debug: print('3> op stack: %s, val stack: %s, node stack: %s'% ( self.op_stack, self.val_stack, self.node_stack))
        ret = self.cleanup_stacks()

        if self.debug: print('4> op stack: %s, val stack: %s, node stack: %s'% ( self.op_stack, self.val_stack, self.node_stack))
        if self.debug: print('<<<<< Depth: %s, returning: %s'% (ExprParser.depth, ret))
        ExprParser.depth = ExprParser.depth - 1
        if self.debug: print('')
        return ret

    def parse(self):
        """
        Tokenizes and parses an arithmetic expression into a parse tree.

        @return: Returns a token string.
        @rtype: lems.parser.expr.ExprNode
        """
        #print("Parsing: %s"%self.parse_string)
        self.tokenize()
        if self.debug: print("Tokens found: %s"%self.token_list)

        try:
            parse_tree = self.parse2()
        except Exception as e:
                raise e
        return parse_tree

    def parse2(self):
        self.op_stack = Stack()
        self.val_stack = Stack()
        self.node_stack = Stack()

        self.op_stack.push('$')
        self.val_stack.push('$')
        try:
            ret = self.parse_token_list_rec(self.op_priority['$'])
        except Exception as e:
            raise e
    
        return ret

    def __str__(self):
        return str(self.token_list)
