
# class PlayState(State):

#     pause_clicked: Shared = Ref("pause_clicked")
#     stop_clicked: Shared = Ref("stop_clicked")

#     def __init__(self, stopped: State, paused: State, store: Storage, name: str):
#         super().__init__(store, name)
#         self._stopped = stopped
#         self._paused = paused

#     def enter(self):
#         # self.play
#         pass

#     def update(self) -> State:
#         if self.pause_clicked.value is True:
#             return self._paused
#         if self.stop_clicked.value is True:
#             return self._stopped
#         if self.finished is True:
#             return self._stopped
#         return self