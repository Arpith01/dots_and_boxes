import numpy as np
from player import Player
from collections import defaultdict


class Game:
    player1_score = 0
    player2_score = 0
    winner_id = 0
    game_over = False
    box_closed = False
    invalid_move = False
    two_boxes_closed = False
    filled_edges = []

    def __init__(self, height, width, player1, player2, invalid_move_reward=-20, box_closed_reward=1, winning_reward=4, losing_reward=-4, tie_reward=0):
        self.height = height
        self.width = width
        self.board = np.zeros((height, width))
        self.player1 = player1
        self.player2 = player2
        self.player1.id = 1
        self.player2.id = -1
        self.active_player_id = 1
        self.invalid_move_reward = invalid_move_reward
        self.box_closed_reward = box_closed_reward
        self.double_closed_reward = box_closed_reward*2
        self.winning_reward = winning_reward
        self.losing_reward = losing_reward
        self.tie_reward = tie_reward
        self.create_edge_dictionary()
        self.init_state = self.get_state()

    @property
    def active_player(self):
        return self.get_player_with_id(self.active_player_id)

    @property
    def inactive_player(self):
        return self.get_player_with_id(self.active_player.id*-1)

    def get_player_with_id(self, id):
        if(id == self.player1.id):
            return self.player1
        else:
            return self.player2

    def reset_turn_params(self):
        self.invalid_move = False
        self.box_closed = False
        self.two_boxes_closed = False

    def play(self, edge_index):
        self.reset_turn_params()
        self.update_grid(edge_index)
        status = self.get_status()
        return status

    def update_scores(self):
        flatboard = self.board.flatten()
        self.player1_score = len(
            list(filter(lambda x: x == self.player1.id, flatboard)))
        self.player2_score = len(
            list(filter(lambda x: x == self.player2.id, flatboard)))

    def get_status(self):
        self.update_scores()
        if(not(np.any((self.board == 0)))):
            self.game_over = True
            if(self.player1_score == self.player2_score):
                self.winner_id = 0
            elif(self.player1_score > self.player2_score):
                self.winner_id = self.player1.id
            else:
                self.winner_id = self.player2.id
        status = {}
        status["game_over"] = self.game_over
        status["winner"] = self.winner_id
        status["invalid_move"] = self.invalid_move
        status["box_closed"] = self.box_closed
        status["double_closed"] = self.two_boxes_closed
        return status

    def switch_player(self):
        if(not(self.invalid_move or self.box_closed or self.two_boxes_closed)):
            self.active_player_id *= -1

    def update_grid(self, edge_index):
        if(self.edges[edge_index] != 0):
            self.invalid_move = True
            return -10
        edge = self.index_to_coordinates[edge_index]
        self.filled_edges.append(edge)
        self.edges[edge_index] = self.active_player_id
        self.check_box_closed(edge)

    def get_state(self):
        return np.expand_dims(np.append(self.edges, self.board.flatten()), axis=1).T

    def check_box_closed(self, edge):
        is_horizontal = edge[0][1] == edge[1][1]
        if(is_horizontal):
            upper_edges = []
            lower_edges = []
            upper_edge_indexes = []
            lower_edge_indexes = []
            if(edge[0][1] != 0):
                upper_edge = tuple(
                    sorted([(edge[0][0], edge[0][1] - 1), (edge[1][0], edge[1][1] - 1)]))
                upper_edges.append(upper_edge)
                upper_left_edge = tuple(
                    sorted([(edge[0][0], edge[0][1] - 1), (edge[0][0], edge[0][1])]))
                upper_edges.append(upper_left_edge)
                upper_right_edge = tuple(
                    sorted([(edge[1][0], edge[0][1] - 1), (edge[1][0], edge[1][1])]))
                upper_edges.append(upper_right_edge)
                upper_edge_index = self.coordinates_to_index[upper_edge]
                upper_left_edge_index = self.coordinates_to_index[upper_left_edge]
                upper_right_edge_index = self.coordinates_to_index[upper_right_edge]
                upper_edge_indexes = [
                    upper_edge_index, upper_left_edge_index, upper_right_edge_index]

            if(edge[0][1] != self.height):
                lower_edge = tuple(
                    sorted([(edge[0][0], edge[0][1] + 1), (edge[1][0], edge[1][1] + 1)]))
                lower_edges.append(lower_edge)
                lower_left_edge = tuple(
                    sorted([(edge[0][0], edge[0][1]), (edge[0][0], edge[1][1] + 1)]))
                lower_edges.append(lower_left_edge)
                lower_right_edge = tuple(
                    sorted([(edge[1][0], edge[0][1]), (edge[1][0], edge[1][1] + 1)]))
                lower_edges.append(lower_right_edge)
                lower_edge_index = self.coordinates_to_index[lower_edge]
                lower_left_edge_index = self.coordinates_to_index[lower_left_edge]
                lower_right_edge_index = self.coordinates_to_index[lower_right_edge]
                lower_edge_indexes = [
                    lower_edge_index, lower_left_edge_index, lower_right_edge_index]

            upper_box_closed = False
            for i in upper_edge_indexes:
                if(self.edges[i] != 0):
                    upper_box_closed = True
                else:
                    upper_box_closed = False
                    break
            if(upper_box_closed):
                # print("Upper box closed")
                upperBoxCorner = (edge[0][0], edge[0][1]-1)
                self.board[upperBoxCorner[::-1]] = self.active_player_id
                self.box_closed = True

            lower_box_closed = False
            for i in lower_edge_indexes:
                if(self.edges[i] != 0):
                    lower_box_closed = True
                else:
                    lower_box_closed = False
                    break
            if(lower_box_closed):
                # print("lower box closed")
                lowerBoxTopCorner = edge[0]
                self.board[lowerBoxTopCorner[::-1]] = self.active_player_id
                self.box_closed = True
            self.two_boxes_closed = lower_box_closed and upper_box_closed
        else:
            left_edges = []
            right_edges = []
            left_edge_indexes = []
            right_edge_indexes = []
            if(edge[0][0] != 0):
                left_edge = tuple(
                    sorted([(edge[0][0]-1, edge[0][1]), (edge[1][0]-1, edge[1][1])]))
                left_edges.append(left_edge)
                left_upper_edge = tuple(
                    sorted([(edge[0][0]-1, edge[0][1]), (edge[0][0], edge[0][1])]))
                left_edges.append(left_upper_edge)
                left_lower_edge = tuple(
                    sorted([(edge[0][0] - 1, edge[1][1]), (edge[0][0], edge[1][1])]))
                left_edges.append(left_lower_edge)

                left_edge_index = self.coordinates_to_index[left_edge]
                left_upper_edge_index = self.coordinates_to_index[left_upper_edge]
                left_lower_edge_index = self.coordinates_to_index[left_lower_edge]
                left_edge_indexes = [left_edge_index,
                                     left_upper_edge_index, left_lower_edge_index]

            if(edge[0][0] != self.width):
                right_edge = tuple(
                    sorted([(edge[0][0]+1, edge[0][1]), (edge[1][0]+1, edge[1][1])]))
                right_edges.append(right_edge)
                right_upper_edge = tuple(
                    sorted([(edge[0][0], edge[0][1]), (edge[0][0] + 1, edge[0][1])]))
                right_edges.append(right_upper_edge)
                right_lower_edge = tuple(
                    sorted([(edge[0][0], edge[1][1]), (edge[1][0] + 1, edge[1][1])]))
                right_edges.append(right_lower_edge)

                right_edge_index = self.coordinates_to_index[right_edge]
                right_upper_edge_index = self.coordinates_to_index[right_upper_edge]
                right_lower_edge_index = self.coordinates_to_index[right_lower_edge]
                right_edge_indexes = [
                    right_edge_index, right_upper_edge_index, right_lower_edge_index]

            left_box_closed = False
            for i in left_edge_indexes:
                if(self.edges[i] != 0):
                    left_box_closed = True
                else:
                    left_box_closed = False
                    break
            if(left_box_closed):
                # print("left box closed")
                left_box_top_corner = (edge[0][0]-1, edge[0][1])
                self.board[left_box_top_corner[::-1]] = self.active_player_id
                self.box_closed = True

            right_box_closed = False
            for i in right_edge_indexes:
                if(self.edges[i] != 0):
                    right_box_closed = True
                else:
                    right_box_closed = False
                    break
            if(right_box_closed):
                # print("right box closed")
                right_box_top_corner = edge[0]
                self.board[right_box_top_corner[::-1]] = self.active_player_id
                self.box_closed = True
            self.two_boxes_closed = left_box_closed and right_box_closed

    def create_edge_dictionary(self):
        self.index_to_coordinates = {}
        self.coordinates_to_index = defaultdict(int)
        counter = 0
        for y in range(self.height+1):
            for x in range(self.width):
                coords = tuple(sorted([(x, y), (x+1, y)]))
                self.index_to_coordinates[counter] = coords
                self.coordinates_to_index[coords] = counter
                counter += 1
            
            if(y != self.height):
                for x in range(self.width+1):
                    coords = tuple(sorted([(x, y), (x, y+1)]))
                    self.index_to_coordinates[counter] = coords
                    self.coordinates_to_index[coords] = counter
                    counter += 1

        self.edges = np.zeros((counter, 1), dtype="int")

    def draw_game(self, scale=1):
        h_between = scale * 5
        v_between = scale * 5
        corner_chars = "+ "
        double_spaces = "  "
        edge_chars = ". "
        active_edge_chars = "* "
        numbered_edge = lambda i:str(i).ljust(2, " ")
        number_of_lines = (self.height+1) + (self.height*v_between)
        number_of_chars_edge = (self.width + 1) + (self.width*h_between)
        box_number = -1
        for line_number in range(number_of_lines):
            string_to_print = ""
            if(line_number % (v_between+1) == 0):
                # Current line is a horizontal edge of boxes
                for char_i in range(number_of_chars_edge):
                    y0 = line_number / (v_between+1)
                    x0 = char_i // (h_between+1)
                    x1 = x0 + 1
                    edge_number = self.coordinates_to_index[tuple(sorted([(x0, y0), (x1, y0)]))]
                    if(char_i % (h_between+1) == 0):
                        # Cursor is corner of a box
                        string_to_print +=  corner_chars
                    elif((char_i -(h_between+1)//2) % (h_between+1) == 0):
                        # Cursor is middle of an edge of a box
                        string_to_print += numbered_edge(edge_number)
                    else:
                        # Other characters of the edges
                        if(self.edges[edge_number] != 0):
                            # Current edge is already used by some player
                            string_to_print += active_edge_chars
                        else:
                            string_to_print += edge_chars
            elif((line_number-((v_between+1)//2))%(v_between+1)==0):
                # Current line passes throught center of boxes
                for char_i in range(number_of_chars_edge):
                    x0 = char_i/(h_between+1)
                    y0 = line_number // (v_between+1)
                    y1 = y0 + 1
                    edge_number = self.coordinates_to_index[tuple(sorted([(x0, y0), (x0, y1)]))]
                    if(char_i % (h_between+1) == 0):
                        # Cursor is on the vertical edge of a box
                        string_to_print += numbered_edge(edge_number)
                    elif((char_i -(h_between+1)//2) % (h_between+1) == 0):
                        # Cursor is at the center of the box.
                        box_number+=1
                        string_to_print += numbered_edge(box_number)
                    else:
                        string_to_print += double_spaces
            else:
                # Current line passes through the rest of the points of the boxes other than edges and centers
                for char_i in range(number_of_chars_edge):
                    x0 = char_i/(h_between+1)
                    y0 = line_number // (v_between+1)
                    y1 = y0 + 1
                    edge_number = self.coordinates_to_index[tuple(sorted([(x0, y0), (x0, y1)]))]
                    if(char_i % (h_between+1) == 0):
                        # Cursor on the edge of a box
                        if(self.edges[edge_number] != 0):
                            # Cursor on an active edge 
                            string_to_print += active_edge_chars
                        else:
                            # Cursor on a non-active edge
                            string_to_print += edge_chars
                    else:
                        string_to_print += double_spaces

            print(string_to_print)


if __name__ == "__main__":
    player1 = Player()
    player1.name = "player1"
    player2 = Player()
    player2.name = "player2"
    game = Game(2, 2, player1, player2)
    game.draw_game()
    print(game.play(8))
    game.draw_game()
