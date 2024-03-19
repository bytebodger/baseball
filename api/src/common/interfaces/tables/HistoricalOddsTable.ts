import type { Team } from '../../enums/Team.js';

export interface HistoricalOddsTable {
   date: string,
   historical_odds_id: number,
   host_moneyline: number | null,
   host_score: number,
   host_team: Team,
   over_moneyline: number | null,
   over_under: number | null,
   season: number,
   under_moneyline: number | null,
   visitor_moneyline: number | null,
   visitor_score: number,
   visitor_team: Team,
}