#pragma once
// The sounds of the game, synthesised when the program starts (no asset
// files). Table tennis is played by ear as much as by eye: the click off the
// table and the knock off the bat give the rhythm of a rally, and how loud
// and sharp they are says how hard the ball was hit. So each sound is played
// louder and a little higher for a faster ball, and placed left or right
// according to where it happened.

#include <memory>

namespace pp {

class Sounds {
 public:
  Sounds();
  ~Sounds();
  Sounds(const Sounds&) = delete;
  Sounds& operator=(const Sounds&) = delete;

  // `speed`: the ball's speed in m/s. `side`: -1 far left of the table, +1 far
  // right. `nearness`: 1 at the player's end, 0 at the opponent's (further
  // away is quieter).
  void bat(float speed, float side, float nearness);
  void table(float speed, float side, float nearness);
  void net(float speed);
  void floor(float speed, float side);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pp
