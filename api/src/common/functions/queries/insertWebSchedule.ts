import { Table } from '../../enums/Table.js';
import { insertIntoDBTable } from './insertIntoDBTable.js';

interface Fields {
   has_been_played: boolean,
   html: string,
   season: number,
   time_checked: number,
   time_processed: number | null,
   time_retrieved: number,
   url: string,
}

export const insertWebSchedule = async (fields: Fields) => {
   return await insertIntoDBTable(Table.webSchedule, fields);
}