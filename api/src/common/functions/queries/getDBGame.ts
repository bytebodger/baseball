import { dbClient } from '../../constants/dbClient.js';

export const getDBGame = async (baseballReferenceId: string) => {
   return await dbClient.query(
      `
         SELECT
            game.baseball_reference_id
            ,game.day_of_year
            ,game.game_id
            ,game.game_of_season
            ,game.home_plate_umpire
            ,game.host_moneyline
            ,game.host_score
            ,game.host_team_id
            ,game.hour_of_day
            ,game.over_moneyline
            ,game.over_under
            ,game.playing_surface
            ,game.season
            ,game.temperature
            ,game.under_moneyline
            ,game.venue
            ,game.visitor_moneyline
            ,game.visitor_score
            ,game.visitor_team_id
         FROM
            game
         WHERE
             game.baseball_reference_id = $1
      `,
      [
         baseballReferenceId,
      ],
   )
}