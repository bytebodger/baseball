import { dbClient } from '../../constants/dbClient.js';
import type { Team } from '../../enums/Team.js';

export const getDBHistoricalOdds = async (season: number, date: string, visitor: Team, host: Team) => {
   return await dbClient.query(
      `
         SELECT
            historical_odds.date
            ,historical_odds.historical_odds_id
            ,historical_odds.host_moneyline
            ,historical_odds.host_score
            ,historical_odds.host_team
            ,historical_odds.over_moneyline
            ,historical_odds.over_under
            ,historical_odds.season
            ,historical_odds.under_moneyline
            ,historical_odds.visitor_moneyline
            ,historical_odds.visitor_score
            ,historical_odds.visitor_team
         FROM
            historical_odds
         WHERE
            historical_odds.season = $1 
         AND
            historical_odds.date = $2
         AND
            historical_odds.visitor_team = $3
         AND
            historical_odds.host_team = $4
      `,
      [
         season,
         date,
         visitor,
         host,
      ],
   )
}