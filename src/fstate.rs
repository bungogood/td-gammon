use bkgm::{Dice, GameState, Position, State};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FState<G: State> {
    pub state: G,
    pub turn: bool,
}

impl<G: State + Send> State for FState<G> {
    const NUM_CHECKERS: u8 = G::NUM_CHECKERS;

    fn new() -> Self {
        Self {
            state: G::new(),
            turn: true,
        }
    }

    fn position(&self) -> Position {
        self.state.position()
    }

    fn flip(&self) -> Self {
        Self {
            state: self.state.flip(),
            turn: !self.turn,
        }
    }

    fn game_state(&self) -> GameState {
        self.state.game_state()
    }

    fn possible_positions(&self, dice: &Dice) -> Vec<Self> {
        self.state
            .possible_positions(dice)
            .iter()
            .map(|pos| FState {
                state: *pos,
                turn: !self.turn,
            })
            .collect()
    }

    fn from_position(position: Position) -> Self {
        Self {
            state: G::from_position(position),
            turn: true,
        }
    }

    fn x_bar(&self) -> u8 {
        self.state.x_bar()
    }

    fn o_bar(&self) -> u8 {
        self.state.o_bar()
    }

    fn x_off(&self) -> u8 {
        self.state.x_off()
    }

    fn o_off(&self) -> u8 {
        self.state.o_off()
    }

    fn pip(&self, pip: usize) -> i8 {
        self.state.pip(pip)
    }

    fn board(&self) -> [i8; 24] {
        self.state.board()
    }

    fn dbhash(&self) -> usize {
        self.state.dbhash()
    }
}

impl<G: State> FState<G> {
    pub fn f_game_state(&self) -> bkgm::GameState {
        if self.turn {
            self.state.game_state()
        } else {
            self.state.flip().game_state()
        }
    }

    pub fn f_state(&self) -> G {
        if self.turn {
            self.state
        } else {
            self.state.flip()
        }
    }
}
